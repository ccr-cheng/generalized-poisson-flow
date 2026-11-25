"""Download indexed AFDB v4 structures from the Steinegger Foldcomp mirror.

Cache lookup/index files, then fetch bounded byte ranges and decompress selected
records. Auto mode tries the representative database (which covers d_FS), then
the full v4 database for remaining IDs. Full-v4 metadata needs about 14 GiB;
neither mode downloads the entire structure database as a single file.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import http.client
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


MIRROR = "https://opendata.mmseqs.org/foldcomp"
DATABASES = ("afdb_rep_v4", "afdb_uniprot_v4")
IDENTIFIER = re.compile(r"AF-[A-Za-z0-9]+(?:-\d+)?-F[1-9]\d*-model_v4\Z")


class DownloadError(RuntimeError):
    pass


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def size_string(size):
    return f"{size / (1024 ** 3):.3f} GiB"


class MirrorClient:
    def __init__(self, timeout=60, retries=3):
        self.timeout, self.retries = timeout, retries

    def request(self, url, method="GET", headers=None):
        request = urllib.request.Request(url, method=method, headers={
            "User-Agent": "GPFlow-AFDB-v4-downloader/1.0", "Accept-Encoding": "identity", **(headers or {}),
        })
        return urllib.request.urlopen(request, timeout=self.timeout)

    def retry(self, operation):
        for attempt in range(self.retries + 1):
            try:
                return operation()
            except DownloadError:
                raise
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == self.retries:
                    raise
            except (OSError, EOFError, http.client.HTTPException):
                if attempt == self.retries:
                    raise
            time.sleep(min(2 ** attempt, 8))

    def describe(self, url):
        def head():
            with self.request(url, method="HEAD") as response:
                size = response.headers.get("Content-Length")
                if response.status != 200 or size is None:
                    raise DownloadError(f"Missing file size from {url}")
                return {"url": url, "size": int(size), "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified")}
        return self.retry(head)

    @staticmethod
    def check_range(response, start, end, total):
        expected = f"bytes {start}-{end}/{total}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            # Never consume a full response: the structure blob can exceed 1 TB.
            raise DownloadError(
                f"Server did not honor byte range {expected}: HTTP {response.status}, "
                f"Content-Range={response.headers.get('Content-Range')!r}"
            )

    def metadata_file(self, info, path):
        stamp = path.with_name(path.name + ".remote.json")
        old = json.loads(stamp.read_text()) if stamp.exists() else None
        if path.is_file() and path.stat().st_size == info["size"] and old == info:
            return path
        partial = path.with_name(path.name + ".part")
        partial_stamp = path.with_name(path.name + ".part.remote.json")
        partial_info = json.loads(partial_stamp.read_text()) if partial_stamp.exists() else None
        if partial_info != info:
            partial.unlink(missing_ok=True)
        atomic_json(partial_stamp, info)
        print(f"Caching {path.name}: {size_string(info['size'])}", flush=True)

        def publish():
            partial.replace(path)
            atomic_json(stamp, info)
            partial_stamp.unlink(missing_ok=True)
            return path

        def download():
            start = partial.stat().st_size if partial.exists() else 0
            if start > info["size"]:
                partial.unlink()
                start = 0
            if start == info["size"]:
                return publish()
            headers = {"Range": f"bytes={start}-{info['size'] - 1}"} if start else {}
            if start and info["etag"]:
                headers["If-Range"] = info["etag"]
            with self.request(info["url"], headers=headers) as response:
                if start:
                    self.check_range(response, start, info["size"] - 1, info["size"])
                elif response.status != 200:
                    raise DownloadError(f"Unexpected metadata response: HTTP {response.status}")
                if info["etag"] and response.headers.get("ETag") != info["etag"]:
                    raise DownloadError(f"Mirror metadata changed while downloading {path.name}; rerun")
                last_update = time.monotonic()
                with partial.open("ab" if start else "wb") as target:
                    total = start
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                        total += len(chunk)
                        if total > info["size"]:
                            raise DownloadError(f"Metadata exceeded the advertised size: {path.name}")
                        if time.monotonic() - last_update > 10:
                            print(f"  {path.name}: {size_string(total)} / {size_string(info['size'])}", flush=True)
                            last_update = time.monotonic()
                if total != info["size"]:
                    raise EOFError(f"Incomplete metadata: {path.name}, {total}/{info['size']} bytes")
            return publish()
        return self.retry(download)

    def read_range(self, info, start, end):
        if not 0 <= start <= end < info["size"]:
            raise DownloadError(f"Invalid byte range {start}-{end} for {info['url']}")
        def download():
            headers = {"Range": f"bytes={start}-{end}"}
            if info["etag"]:
                headers["If-Range"] = info["etag"]
            with self.request(info["url"], headers=headers) as response:
                self.check_range(response, start, end, info["size"])
                if info["etag"] and response.headers.get("ETag") != info["etag"]:
                    raise DownloadError("Mirror structure database changed during download; rerun")
                expected = end - start + 1
                data = response.read(expected + 1)
                if len(data) != expected:
                    raise EOFError(f"Incomplete range: received {len(data)} of {expected} bytes")
                return data
        return self.retry(download)


@dataclass(frozen=True)
class Entry:
    name: str
    offset: int
    length: int


def read_ids(path):
    names = set()
    with path.open() as source:
        for number, line in enumerate(source, 1):
            name = line.strip()
            if not name:
                continue
            if not IDENTIFIER.fullmatch(name):
                raise DownloadError(f"{path}:{number}: expected AF-<accession>-F<n>-model_v4, got {name!r}")
            names.add(name)
    if not names:
        raise DownloadError("The input index is empty")
    return names


def valid_pdb(path):
    if not path.is_file():
        return False
    # A partial old download can contain valid atoms without a complete chain.
    with path.open("rb") as source:
        source.seek(0, 2)
        source.seek(max(0, source.tell() - 4096))
        tail = source.read().splitlines()
    nonempty = [line for line in tail if line.strip()]
    if not nonempty or nonempty[-1][:6].strip() not in (b"TER", b"END", b"ENDMDL"):
        return False
    # Also rejects HTML/JSON errors saved by older download scripts.
    with path.open(errors="replace") as source:
        return any(line.startswith("ATOM  ") and line[12:16].strip() == "CA" for line in source)


def resolve_entries(client, mirror, database, wanted, cache):
    base = f"{mirror}/{database}"
    info = {suffix: client.describe(base + suffix) for suffix in (".lookup", ".index", "")}
    cache.mkdir(parents=True, exist_ok=True)
    lookup = client.metadata_file(info[".lookup"], cache / (database + ".lookup"))
    index = client.metadata_file(info[".index"], cache / (database + ".index"))
    print(f"Matching {len(wanted):,} requested IDs against {database}", flush=True)
    keys = {}
    with lookup.open() as source:
        for line in source:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2 and fields[1] in wanted:
                keys[fields[0]] = fields[1]
    entries = []
    with index.open() as source:
        for line in source:
            fields = line.split()
            if len(fields) == 3 and fields[0] in keys:
                name = keys.pop(fields[0])
                offset, length = int(fields[1]), int(fields[2])
                if offset < 0 or length < 2 or offset + length > info[""]["size"]:
                    raise DownloadError(f"Invalid mirror index record for {name}")
                entries.append(Entry(name, offset, length))
    if keys:
        raise DownloadError(f"{database} lookup/index disagree for {len(keys)} requested IDs")
    entries.sort(key=lambda entry: entry.offset)
    return entries, info[""]


def groups(entries, max_bytes, max_gap):
    batch = []
    end = 0
    for entry in entries:
        if batch and (entry.offset - end > max_gap or entry.offset + entry.length - batch[0].offset > max_bytes):
            yield batch
            batch = []
        batch.append(entry)
        end = max(end if len(batch) > 1 else 0, entry.offset + entry.length)
    if batch:
        yield batch


def decode_and_write(data, entry, destination, foldcomp):
    if not data.startswith(b"FCMP") or not data.endswith(b"\0"):
        raise DownloadError(f"Invalid Foldcomp database record for {entry.name}")
    name, pdb = foldcomp.decompress(data[:-1])
    for suffix in (".cif.gz", ".pdb.gz", ".cif", ".pdb"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    if name != entry.name:
        raise DownloadError(f"Mirror returned {name!r} for requested {entry.name}")
    if not any(line.startswith("ATOM  ") and line[12:16].strip() == "CA" for line in pdb.splitlines()):
        raise DownloadError(f"Decoded structure has no C-alpha atoms: {entry.name}")
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_text(pdb)
    temporary.replace(destination)


def download_entries(client, info, entries, output, args, foldcomp):
    for entry in entries:
        if entry.length > args.max_range_bytes:
            raise DownloadError(
                f"{entry.name} needs {entry.length} bytes, exceeding --max-range-bytes="
                f"{args.max_range_bytes}; increase that limit"
            )
    count, failures = 0, []
    batches = groups(entries, args.max_range_bytes, args.max_gap_bytes)
    planned = sum(max(entry.offset + entry.length for entry in batch) - batch[0].offset
                  for batch in groups(entries, args.max_range_bytes, args.max_gap_bytes))
    print(f"Fetching {len(entries):,} structures in bounded ranges ({size_string(planned)} including read-ahead)", flush=True)

    def download(batch):
        start = batch[0].offset
        end = max(entry.offset + entry.length for entry in batch) - 1
        payload = client.read_range(info, start, end)
        completed, errors = 0, []
        for entry in batch:
            try:
                offset = entry.offset - start
                decode_and_write(payload[offset:offset + entry.length], entry, output / (entry.name + ".pdb"), foldcomp)
                completed += 1
            except Exception as exc:
                errors.append({"id": entry.name, "error": str(exc)})
        return completed, errors

    last_update = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < args.workers * 2:
                batch = next(batches, None)
                if batch is None:
                    exhausted = True
                    break
                pending[pool.submit(download, batch)] = batch
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                batch = pending.pop(future)
                try:
                    completed, errors = future.result()
                    count += completed
                    failures.extend(errors)
                except Exception as exc:
                    failures.extend({"id": entry.name, "error": str(exc)} for entry in batch)
                    if isinstance(exc, DownloadError):
                        for outstanding in pending:
                            outstanding.cancel()
                        raise
            if time.monotonic() - last_update > 10:
                print(f"  Saved {count:,}/{len(entries):,}; failures {len(failures):,}", flush=True)
                last_update = time.monotonic()
    return count, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("index_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--database", choices=("auto", *DATABASES), default="auto",
                        help="auto tries representatives, then full v4 for remaining IDs (metadata is ~14 GiB)")
    parser.add_argument("--mirror-url", default=MIRROR)
    parser.add_argument("--cache-dir", type=Path, help="metadata cache (default: OUTPUT_DIR/.afdb_mirror_cache)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-range-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-gap-bytes", type=int, default=64 * 1024)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0 or args.max_range_bytes < 1 or args.max_gap_bytes < 0:
        parser.error("workers, timeout and max-range-bytes must be positive; retries and max-gap-bytes must be nonnegative")
    try:
        import foldcomp
    except ImportError:
        parser.error("foldcomp is required; run `uv sync --inexact` in the repository or install the pinned project dependencies")

    output = args.output_dir.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cache = (args.cache_dir or output / ".afdb_mirror_cache").resolve()
    mirror = args.mirror_url.rstrip("/")
    cache = cache / hashlib.sha256(mirror.encode()).hexdigest()[:12]
    report = {"index_file": str(args.index_file.resolve()), "mirror": mirror,
              "format": "Foldcomp AFDB v4 (lossy reconstruction)", "databases": [],
              "requested": 0, "skipped_existing": 0, "downloaded": 0, "failures": []}
    try:
        wanted = read_ids(args.index_file)
        report["requested"] = len(wanted)
        pending = {name for name in wanted if not valid_pdb(raw / (name + ".pdb"))}
        report["skipped_existing"] = len(wanted) - len(pending)
        print(f"Requested {len(wanted):,}; valid existing PDBs {report['skipped_existing']:,}", flush=True)
        client = MirrorClient(args.timeout, args.retries)
        databases = DATABASES if args.database == "auto" else (args.database,)
        for database in databases:
            if not pending:
                break
            if database == "afdb_uniprot_v4":
                print("Full v4 lookup/index cache needs about 14 GiB; fetching selected structure ranges only.", flush=True)
            entries, info = resolve_entries(client, mirror, database, pending, cache)
            report["databases"].append({"name": database, "matched": len(entries), "source": info})
            pending.difference_update(entry.name for entry in entries)
            completed, failures = download_entries(client, info, entries, raw, args, foldcomp)
            report["downloaded"] += completed
            report["failures"].extend(failures)
        missing = sorted(pending)
        (output / "missing_ids.txt").write_text("".join(name + "\n" for name in missing))
        report["missing"] = len(missing)
        report["status"] = "SUCCESS" if not missing and not report["failures"] else "INCOMPLETE"
    except (DownloadError, OSError, ValueError, EOFError, http.client.HTTPException) as exc:
        report["status"] = "FAILED"
        report["error"] = str(exc)
    atomic_json(output / "download_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
