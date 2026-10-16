#!/usr/bin/env python3
"""
fasta_tax_lengths.py - sequence-length statistics by taxonomic group.

Reads a FASTA file (plain or gzip-compressed) whose headers look like

    >ACCESSION Rank1;Rank2;Rank3;...;Species

(e.g. SILVA / PR2 / UNITE-style reference databases) and reports length
statistics, length histograms, and sequence look-ups for any taxon name that
appears anywhere in those lineages.

Requirements: Python 3.8+ (standard library only; matplotlib is optional and
only needed for histograms).

The parsed index is cached as a small gzip-compressed TSV so re-runs are fast.
The cache is tied to the FASTA's size and modification time and is rebuilt
automatically if the FASTA changes.  It is written next to the FASTA when that
folder is writable, otherwise to a per-user cache folder (override with
--cache-dir, disable with --no-cache).

Examples
--------
  # Interactive prompt (type 'help' inside)
  python fasta_tax_lengths.py reference.fasta.gz

  # Length stats for one or more taxa
  python fasta_tax_lengths.py reference.fasta -t Fungi "Homo sapiens"

  # Same, but ignore any sequence whose lineage contains 'uncultured'
  python fasta_tax_lengths.py reference.fasta -t Fungi -x uncultured

  # 20 most-represented taxa
  python fasta_tax_lengths.py reference.fasta --top 20

  # Which Fungi sequences are 3400 bp +/- 10?
  python fasta_tax_lengths.py reference.fasta --who Fungi 3400 10

  # Overlaid histograms for two taxa, saved to a file
  python fasta_tax_lengths.py reference.fasta --hist Fungi + Metazoa --save out.png

Put the FASTA path first on the command line (before options that take
several values such as -t, -x, --hist and --who).
"""

import argparse
import gzip
import hashlib
import os
import sys
import tempfile
from collections import defaultdict, namedtuple
from pathlib import Path
from statistics import mean, median, stdev
from types import SimpleNamespace

CACHE_VERSION = 2
CACHE_SUFFIX = ".tax_index.tsv.gz"
DEFAULT_DELIM = ";"

Record = namedtuple("Record", "length accession taxonomy")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def info(msg):
    print(msg, file=sys.stderr)


def to_int(token):
    """int() that also accepts a leading '+/-', '+-' or the plus-minus sign."""
    for prefix in ("\u00b1", "+-"):
        if token.startswith(prefix):
            token = token[len(prefix):]
    return int(token)


def is_int_token(token):
    try:
        to_int(token)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# FASTA parsing
# --------------------------------------------------------------------------
def open_text(path):
    """Open a plain or gzip-compressed text file (detected by magic bytes)."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_fasta(path):
    """Yield (header_without_>, sequence_length) for every record."""
    header = None
    seq_len = 0
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()          # also removes CR from CRLF files
            if line.startswith(">"):
                if header is not None:
                    yield header, seq_len
                header = line[1:]
                seq_len = 0
            elif header is not None:
                seq_len += len(line)
        if header is not None:
            yield header, seq_len


def split_taxonomy(tax_string, delim):
    return [t.strip() for t in tax_string.split(delim) if t.strip()]


def build_records(fasta_path, delim):
    """Parse the FASTA into a list of Record(length, accession, taxonomy)."""
    records = []
    skipped = 0
    for header, length in parse_fasta(fasta_path):
        parts = header.split(None, 1)
        if len(parts) < 2 or not split_taxonomy(parts[1], delim):
            skipped += 1
            continue
        # tabs would corrupt the TSV cache
        records.append(Record(length, parts[0], parts[1].replace("\t", " ")))
    info(f"Parsed {len(records):,} sequences "
         f"({skipped:,} skipped - no taxonomy in header)")
    return records


# --------------------------------------------------------------------------
# Index + cache
# --------------------------------------------------------------------------
class TaxIndex:
    """taxon name -> list of Records whose lineage contains that name."""

    def __init__(self, records, delim):
        self.records = records
        by_taxon = defaultdict(list)
        for rec in records:
            seen = set()
            for taxon in split_taxonomy(rec.taxonomy, delim):
                if taxon not in seen:
                    seen.add(taxon)
                    by_taxon[taxon].append(rec)
        self.by_taxon = dict(by_taxon)
        self._lower = {}
        for name in self.by_taxon:
            self._lower.setdefault(name.lower(), name)

    def __len__(self):
        return len(self.by_taxon)

    def resolve(self, query):
        """Exact match first, then case-insensitive. Returns (name, records)."""
        if query in self.by_taxon:
            return query, self.by_taxon[query]
        name = self._lower.get(query.lower())
        if name is not None:
            return name, self.by_taxon[name]
        return None, None


def user_cache_dir():
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache")
    return Path(base) / "fasta_tax_lengths"


def cache_candidates(fasta_path, cache_dir=None):
    """Ordered list of places to look for / write the cache."""
    fasta = Path(fasta_path).resolve()
    tag = hashlib.sha1(str(fasta).encode("utf-8")).hexdigest()[:10]
    in_cache_dir = lambda d: Path(d).expanduser() / f"{fasta.name}.{tag}{CACHE_SUFFIX}"
    if cache_dir:
        return [in_cache_dir(cache_dir)]
    return [fasta.with_name(fasta.name + CACHE_SUFFIX),
            in_cache_dir(user_cache_dir())]


def fasta_signature(fasta_path, delim):
    st = os.stat(fasta_path)
    return (f"version={CACHE_VERSION}", f"size={st.st_size}",
            f"mtime_ns={st.st_mtime_ns}", f"delim={delim!r}")


def read_cache(path, signature):
    """Return a list of Records, or None if missing / stale / unreadable."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            first = fh.readline().rstrip("\n").split("\t")
            if first[0] != "#fasta_tax_lengths" or tuple(first[1:]) != signature:
                return None
            records = []
            for line in fh:
                acc, length, tax = line.rstrip("\n").split("\t", 2)
                records.append(Record(int(length), acc, tax))
            return records
    except (OSError, EOFError, ValueError):
        return None


def write_cache(path, signature, records):
    """Atomically write the cache; return True on success."""
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write("\t".join(("#fasta_tax_lengths",) + signature) + "\n")
            for r in records:
                fh.write(f"{r.accession}\t{r.length}\t{r.taxonomy}\n")
        os.replace(tmp, path)
        return True
    except OSError:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def load_index(fasta_path, delim=DEFAULT_DELIM, use_cache=True,
               rebuild=False, cache_dir=None):
    if not Path(fasta_path).is_file():
        sys.exit(f"error: FASTA file not found: {fasta_path}")
    signature = fasta_signature(fasta_path, delim)
    candidates = cache_candidates(fasta_path, cache_dir) if use_cache else []

    if use_cache and not rebuild:
        for path in candidates:
            records = read_cache(path, signature)
            if records is not None:
                info(f"Loaded cached index from {path}")
                return TaxIndex(records, delim)

    info(f"Indexing {fasta_path} ...")
    records = build_records(fasta_path, delim)
    if not records:
        sys.exit("error: no sequences with a parseable taxonomy were found. "
                 "Expected headers like '>ACCESSION Rank1;Rank2;...' "
                 "(see --tax-delim).")
    for path in candidates:
        if write_cache(path, signature, records):
            info(f"Index cached to {path}")
            break
    else:
        if use_cache:
            info("Warning: could not write an index cache (read-only location?); "
                 "use --cache-dir to choose one.")
    return TaxIndex(records, delim)


# --------------------------------------------------------------------------
# Exclusions ("not <term>")
# --------------------------------------------------------------------------
def apply_exclusions(records, exclude):
    """Drop records whose lineage contains any term in `exclude` (substring,
    case-insensitive)."""
    if not exclude:
        return records
    terms = [e.lower() for e in exclude]
    return [r for r in records if not any(t in r.taxonomy.lower() for t in terms)]


def parse_exclusions(tokens):
    """Split ['Fungi', 'not', 'uncultured'] -> (['Fungi'], ['uncultured'])."""
    exclude, rest = [], []
    i = 0
    while i < len(tokens):
        if tokens[i].lower() == "not" and i + 1 < len(tokens):
            exclude.append(tokens[i + 1])
            i += 2
        else:
            rest.append(tokens[i])
            i += 1
    return rest, exclude


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def print_stats(taxon, records, cfg):
    n = len(records)
    print(f"\n  Taxon : {taxon}")
    print(f"  Count : {n:,}")
    if n == 0:
        print("  (no sequences left after exclusions)")
        return
    lengths = [r.length for r in records]
    sd = stdev(lengths) if n > 1 else 0.0
    print(f"  Mean  : {mean(lengths):,.1f} {cfg.unit}")
    print(f"  Median: {median(lengths):,.1f} {cfg.unit}")
    print(f"  Stdev : {sd:,.1f} {cfg.unit}")
    print(f"  Min   : {min(lengths):,} {cfg.unit}")
    print(f"  Max   : {max(lengths):,} {cfg.unit}")


def report_taxa(idx, queries, cfg, exclude=None):
    for q in queries:
        key, records = idx.resolve(q)
        if records is None:
            print(f"\n  '{q}' - not found in index")
            continue
        records = apply_exclusions(records, exclude)
        if exclude:
            print(f"\n  (excluding: {', '.join(exclude)})", end="")
        print_stats(key, records, cfg)


def report_top(idx, n, cfg):
    ranked = sorted(idx.by_taxon.items(), key=lambda kv: len(kv[1]), reverse=True)[:n]
    print(f"\n{'Taxon':<50} {'Count':>8} {'Mean (' + cfg.unit + ')':>12} "
          f"{'Median (' + cfg.unit + ')':>14}")
    print("-" * 88)
    for taxon, recs in ranked:
        lengths = [r.length for r in recs]
        print(f"{taxon[:50]:<50} {len(lengths):>8,} {mean(lengths):>12.1f} "
              f"{median(lengths):>14.1f}")


def report_who(idx, taxon_query, target_len, cfg, tol=0, exclude=None):
    """List sequences in a taxon whose length is within `tol` of `target_len`."""
    key, records = idx.resolve(taxon_query)
    if records is None:
        print(f"\n  '{taxon_query}' - not found in index")
        return
    records = apply_exclusions(records, exclude)
    if exclude:
        print(f"\n  (excluding: {', '.join(exclude)})", end="")
    print_stats(key, records, cfg)
    hits = [r for r in records if abs(r.length - target_len) <= tol]
    spec = f"{target_len}" + (f" +/-{tol}" if tol else "") + f" {cfg.unit}"
    if not hits:
        print(f"\n  No sequences at {spec}.")
        return
    print(f"\n  {len(hits):,} sequence(s) at {spec}:")
    print(f"  {'Accession':<30} {'Length':>8}  Taxonomy")
    print("  " + "-" * 90)
    for r in sorted(hits, key=lambda r: r.length):
        print(f"  {r.accession:<30} {r.length:>8}  {r.taxonomy}")


# --------------------------------------------------------------------------
# Histograms
# --------------------------------------------------------------------------
def plot_histogram(datasets, cfg, output=None):
    """datasets: list of (label, lengths). Overlaid, semi-transparent
    histograms on shared bin edges. Saves to `output` if given, otherwise
    opens an interactive window."""
    try:
        import matplotlib
    except ImportError:
        info("matplotlib is not installed (pip install matplotlib); "
             "cannot draw histograms.")
        return
    lo = min(min(l) for _, l in datasets if l)
    hi = max(max(l) for _, l in datasets if l)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    edges = [lo + i * (hi - lo) / cfg.bins for i in range(cfg.bins + 1)]

    if output:      # no pyplot -> works on headless machines
        from matplotlib.figure import Figure
        fig = Figure(figsize=(10, 5))
        ax = fig.subplots()
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
    for label, lengths in datasets:
        ax.hist(lengths, bins=edges, alpha=0.6, label=label, edgecolor="none")
    ax.set_xlabel(f"Sequence length ({cfg.unit})")
    ax.set_ylabel("Count")
    ax.set_title(cfg.title)
    if len(datasets) > 1:
        ax.legend()
    fig.tight_layout()
    if output:
        try:
            fig.savefig(output, dpi=150)
            print(f"  Saved histogram to {output}")
        except OSError as exc:
            info(f"error: could not save histogram: {exc}")
    else:
        plt.show()
        plt.close(fig)


def collect_datasets(idx, text, extra_exclude=None):
    """Parse 'taxonA [not x] + taxonB [not y]' into [(label, lengths), ...]."""
    datasets = []
    for group in text.split("+"):
        toks, exclude = parse_exclusions(group.split())
        exclude = list(exclude) + list(extra_exclude or [])
        taxon = " ".join(toks).strip()
        if not taxon:
            continue
        key, records = idx.resolve(taxon)
        if records is None:
            print(f"\n  '{taxon}' - not found in index")
            continue
        records = apply_exclusions(records, exclude)
        if not records:
            print(f"\n  '{taxon}' - no sequences left after exclusions")
            continue
        label = key + (f" (not {', '.join(exclude)})" if exclude else "")
        datasets.append((label, [r.length for r in records]))
    return datasets


# --------------------------------------------------------------------------
# --who argument parsing (shared by CLI and interactive mode)
# --------------------------------------------------------------------------
def parse_who(tokens):
    """['Fungi', '3400', '10'] -> ('Fungi', 3400, 10); None if malformed."""
    toks = list(tokens)
    nums = []
    while toks and is_int_token(toks[-1]) and len(nums) < 2:
        nums.insert(0, toks.pop())
    taxon = " ".join(toks)
    if not taxon or not nums:
        return None
    return taxon, to_int(nums[0]), (to_int(nums[1]) if len(nums) > 1 else 0)


# --------------------------------------------------------------------------
# Interactive mode
# --------------------------------------------------------------------------
HELP_TEXT = """\
Commands:
  <taxon> [not <term> ...]                      stats for a taxon
  top [N]                                       top N taxa by count (default 20)
  who <taxon> <len> [tol] [not <term> ...]      sequences at a given length (+/- tol)
  hist <taxon> [not <term> ...] [> file.png]    length histogram
  hist <taxon1> + <taxon2> [not <term> ...]     overlaid histograms
  help                                          show this text
  quit
Taxon names are case-insensitive. 'not <term>' drops sequences whose lineage
contains <term> (substring match)."""


def interactive_loop(idx, cfg):
    try:
        import readline  # noqa: F401  (line editing/history where available)
    except ImportError:
        pass
    print(f"\nIndex ready - {len(idx):,} taxon names.")
    print(HELP_TEXT + "\n")
    while True:
        try:
            raw = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()
        if raw.lower() in ("quit", "exit", "q"):
            break
        if raw.lower() in ("help", "?", "h"):
            print(HELP_TEXT)
        elif cmd == "hist":
            rest = parts[1:]
            output = None
            if ">" in rest:
                i = rest.index(">")
                output = rest[i + 1] if i + 1 < len(rest) else None
                rest = rest[:i]
            datasets = collect_datasets(idx, " ".join(rest))
            if datasets:
                plot_histogram(datasets, cfg, output)
        elif cmd == "top":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
            report_top(idx, n, cfg)
        elif cmd == "who" and len(parts) >= 3:
            toks, exclude = parse_exclusions(parts[1:])
            parsed = parse_who(toks)
            if parsed is None:
                print("  Usage: who <taxon> <length> [tolerance] [not <term> ...]")
                continue
            taxon, target, tol = parsed
            report_who(idx, taxon, target, cfg, tol, exclude)
        else:
            toks, exclude = parse_exclusions(parts)
            report_taxa(idx, [" ".join(toks)], cfg, exclude)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Sequence-length statistics by taxonomic group for a FASTA "
                    "file with headers like '>ACCESSION Rank1;Rank2;...;Species'. "
                    "With no query option, starts an interactive prompt.",
        epilog="Put the FASTA path before options that take several values.")
    p.add_argument("fasta", help="FASTA file (plain or .gz)")
    p.add_argument("-t", "--taxon", nargs="+", metavar="NAME",
                   help="taxon name(s) to summarise (quote names containing spaces)")
    p.add_argument("--top", type=int, metavar="N",
                   help="show the N taxa with the most sequences")
    p.add_argument("--who", nargs="+", metavar="ARG",
                   help="list sequences: --who <taxon> <length> [tolerance]")
    p.add_argument("--hist", nargs="+", metavar="NAME",
                   help="histogram for one or more taxa; separate taxa with '+'")
    p.add_argument("-x", "--exclude", nargs="+", action="extend", metavar="TERM",
                   help="exclude sequences whose lineage contains any of these terms")
    p.add_argument("--save", metavar="FILE",
                   help="save the histogram to FILE (png/pdf/svg) instead of showing it")
    p.add_argument("--bins", type=int, default=50, metavar="N",
                   help="histogram bins (default: 50)")
    p.add_argument("--unit", default="bp",
                   help="length unit used in labels, e.g. bp, nt, aa (default: bp)")
    p.add_argument("--title", default="Sequence length distribution",
                   help="histogram title")
    p.add_argument("--tax-delim", default=DEFAULT_DELIM, metavar="CHAR",
                   help=f"separator between ranks in the header (default: '{DEFAULT_DELIM}')")
    p.add_argument("--cache-dir", metavar="DIR",
                   help="store/read the index cache here instead of next to the FASTA")
    p.add_argument("--no-cache", action="store_true",
                   help="do not read or write an index cache")
    p.add_argument("--rebuild", action="store_true",
                   help="ignore any existing cache and rebuild it from the FASTA")
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):    # never crash on odd characters
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    if args.bins < 1:
        sys.exit("error: --bins must be >= 1")
    cfg = SimpleNamespace(unit=args.unit, title=args.title, bins=args.bins)

    idx = load_index(args.fasta, delim=args.tax_delim,
                     use_cache=not args.no_cache, rebuild=args.rebuild,
                     cache_dir=args.cache_dir)
    info(f"Index ready: {len(idx):,} unique taxon names\n")

    if args.taxon:
        report_taxa(idx, args.taxon, cfg, args.exclude)
    elif args.hist:
        datasets = collect_datasets(idx, " ".join(args.hist), args.exclude)
        if datasets:
            plot_histogram(datasets, cfg, output=args.save)
    elif args.top:
        report_top(idx, args.top, cfg)
    elif args.who:
        parsed = parse_who(args.who)
        if parsed is None:
            sys.exit("Usage: --who <taxon> <length> [tolerance]")
        taxon, target, tol = parsed
        report_who(idx, taxon, target, cfg, tol, args.exclude)
    else:
        interactive_loop(idx, cfg)


if __name__ == "__main__":
    main()
