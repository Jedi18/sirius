#!/usr/bin/env python3
"""Run the 12 unique projection-fold candidate TPC-H queries on GPU.

Standalone script — queries are embedded below (from projection_fold_candidates.tsv).
Does not depend on any other script in this repo.

Prerequisites on the GPU machine:
  pixi run make
  ulimit -l unlimited
  export SIRIUS_CONFIG_FILE=test/cpp/integration/integration.yaml
  export SIRIUS_INTEGRATION_TEST_DB_PATH=/path/to/tpch_sf1.duckdb

Usage:
  python3 scripts/run_projection_fold_queries.py
  python3 scripts/run_projection_fold_queries.py --dry-run
  python3 scripts/run_projection_fold_queries.py --log-dir /tmp/sirius_fold
  python3 scripts/run_projection_fold_queries.py --duckdb-path /data/tpch_sf1.duckdb
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DUCKDB = REPO_ROOT / "build/release/duckdb"
DEFAULT_CONFIG = REPO_ROOT / "test/cpp/integration/integration.yaml"

# 12 unique queries (TPC-H Q1,Q2,Q5,Q7,Q9,Q11,Q13,Q16,Q17,Q19,Q21,Q22).
# Extracted from projection_fold_candidates.tsv — duplicates removed.
QUERIES: list[tuple[str, str]] = [
    (
        "q1",
        "select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, "
        "sum(l_extendedprice) as sum_base_price, "
        "sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, "
        "sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, "
        "avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, "
        "avg(l_discount) as avg_disc, count(*) as count_order "
        "from lineitem where l_shipdate <= date '1995-08-19' "
        "group by l_returnflag, l_linestatus "
        "order by l_returnflag, l_linestatus;",
    ),
    (
        "q2",
        "select s.s_acctbal, s.s_name, n.n_name, p.p_partkey, p.p_mfgr, "
        "s.s_address, s.s_phone, s.s_comment "
        "from part p, supplier s, partsupp ps, nation n, region r "
        "where p.p_partkey = ps.ps_partkey and s.s_suppkey = ps.ps_suppkey "
        "and p.p_size = 41 and p.p_type like '%NICKEL' "
        "and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey "
        "and r.r_name = 'EUROPE' "
        "and ps.ps_supplycost = ( "
        "  select min(ps.ps_supplycost) from partsupp ps, supplier s, nation n, region r "
        "  where p.p_partkey = ps.ps_partkey and s.s_suppkey = ps.ps_suppkey "
        "  and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey "
        "  and r.r_name = 'EUROPE') "
        "order by s.s_acctbal desc, n.n_name, s.s_name, p.p_partkey limit 100;",
    ),
    (
        "q5",
        "select supp_nation, cust_nation, l_year, sum(volume) as revenue from ( "
        "  select n1.n_name as supp_nation, n2.n_name as cust_nation, "
        "  extract(year from l.l_shipdate) as l_year, "
        "  l.l_extendedprice * (1 - l.l_discount) as volume "
        "  from supplier s, lineitem l, orders o, customer c, nation n1, nation n2 "
        "  where s.s_suppkey = l.l_suppkey and o.o_orderkey = l.l_orderkey "
        "  and c.c_custkey = o.o_custkey and s.s_nationkey = n1.n_nationkey "
        "  and c.c_nationkey = n2.n_nationkey "
        "  and ((n1.n_name = 'EGYPT' and n2.n_name = 'UNITED STATES') "
        "    or (n1.n_name = 'UNITED STATES' and n2.n_name = 'EGYPT')) "
        "  and l.l_shipdate between date '1995-01-01' and date '1996-12-31'"
        ") as shipping "
        "group by supp_nation, cust_nation, l_year "
        "order by supp_nation, cust_nation, l_year;",
    ),
    (
        "q7",
        "select o_year, sum(case when nation = 'EGYPT' then volume else 0 end) / sum(volume) as mkt_share "
        "from ( "
        "  select extract(year from o.o_orderdate) as o_year, "
        "  l.l_extendedprice * (1 - l.l_discount) as volume, n2.n_name as nation "
        "  from lineitem l, part p, supplier s, orders o, customer c, nation n1, nation n2, region r "
        "  where p.p_partkey = l.l_partkey and s.s_suppkey = l.l_suppkey "
        "  and l.l_orderkey = o.o_orderkey and o.o_custkey = c.c_custkey "
        "  and c.c_nationkey = n1.n_nationkey and n1.n_regionkey = r.r_regionkey "
        "  and r.r_name = 'MIDDLE EAST' and s.s_nationkey = n2.n_nationkey "
        "  and o.o_orderdate between date '1995-01-01' and date '1996-12-31' "
        "  and p.p_type = 'PROMO BRUSHED COPPER'"
        ") as all_nations group by o_year order by o_year;",
    ),
    (
        "q9",
        "select nation, o_year, sum(amount) as sum_profit from ( "
        "  select n.n_name as nation, extract(year from o.o_orderdate) as o_year, "
        "  l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity as amount "
        "  from part p, supplier s, lineitem l, partsupp ps, orders o, nation n "
        "  where s.s_suppkey = l.l_suppkey and ps.ps_suppkey = l.l_suppkey "
        "  and ps.ps_partkey = l.l_partkey and p.p_partkey = l.l_partkey "
        "  and o.o_orderkey = l.l_orderkey and s.s_nationkey = n.n_nationkey "
        "  and p.p_name like '%yellow%'"
        ") as profit group by nation, o_year order by nation, o_year desc;",
    ),
    (
        "q11",
        "select ps.ps_partkey, sum(ps.ps_supplycost * ps.ps_availqty) as value "
        "from partsupp ps, supplier s, nation n "
        "where ps.ps_suppkey = s.s_suppkey and s.s_nationkey = n.n_nationkey and n.n_name = 'JAPAN' "
        "group by ps.ps_partkey "
        "having sum(ps.ps_supplycost * ps.ps_availqty) > ( "
        "  select sum(ps.ps_supplycost * ps.ps_availqty) * 0.0001000000 "
        "  from partsupp ps, supplier s, nation n "
        "  where ps.ps_suppkey = s.s_suppkey and s.s_nationkey = n.n_nationkey and n.n_name = 'JAPAN') "
        "order by value desc;",
    ),
    (
        "q13",
        "select c_count, count(*) as custdist from ( "
        "  select c.c_custkey, count(o.o_orderkey) "
        "  from customer c left outer join orders o "
        "  on c.c_custkey = o.o_custkey and o.o_comment not like '%special%requests%' "
        "  group by c.c_custkey"
        ") as orders (c_custkey, c_count) "
        "group by c_count order by custdist desc, c_count desc;",
    ),
    (
        "q16",
        "select p.p_brand, p.p_type, p.p_size, count(distinct ps.ps_suppkey) as supplier_cnt "
        "from partsupp ps, part p "
        "where p.p_partkey = ps.ps_partkey and p.p_brand <> 'Brand#21' "
        "and p.p_type not like 'MEDIUM PLATED%' "
        "and p.p_size in (38, 2, 8, 31, 44, 5, 14, 24) "
        "and ps.ps_suppkey not in ( "
        "  select s.s_suppkey from supplier s where s.s_comment like '%Customer%Complaints%') "
        "group by p.p_brand, p.p_type, p.p_size "
        "order by supplier_cnt desc, p.p_brand, p.p_type, p.p_size;",
    ),
    (
        "q17",
        "select sum(l.l_extendedprice) / 7.0 as avg_yearly "
        "from lineitem l, part p "
        "where p.p_partkey = l.l_partkey and p.p_brand = 'Brand#13' "
        "and p.p_container = 'JUMBO CAN' "
        "and l.l_quantity < ( "
        "  select 0.2 * avg(l2.l_quantity) from lineitem l2 where l2.l_partkey = p.p_partkey);",
    ),
    (
        "q19",
        "select sum(l.l_extendedprice* (1 - l.l_discount)) as revenue "
        "from lineitem l, part p where ( "
        "  p.p_partkey = l.l_partkey and p.p_brand = 'Brand#41' "
        "  and p.p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') "
        "  and l.l_quantity >= 2 and l.l_quantity <= 2 + 10 "
        "  and p.p_size between 1 and 5 "
        "  and l.l_shipmode in ('AIR', 'AIR REG') and l.l_shipinstruct = 'DELIVER IN PERSON') or ( "
        "  p.p_partkey = l.l_partkey and p.p_brand = 'Brand#13' "
        "  and p.p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') "
        "  and l.l_quantity >= 14 and l.l_quantity <= 14 + 10 "
        "  and p.p_size between 1 and 10 "
        "  and l.l_shipmode in ('AIR', 'AIR REG') and l.l_shipinstruct = 'DELIVER IN PERSON') or ( "
        "  p.p_partkey = l.l_partkey and p.p_brand = 'Brand#55' "
        "  and p.p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') "
        "  and l.l_quantity >= 23 and l.l_quantity <= 23 + 10 "
        "  and p.p_size between 1 and 15 "
        "  and l.l_shipmode in ('AIR', 'AIR REG') and l.l_shipinstruct = 'DELIVER IN PERSON');",
    ),
    (
        "q21",
        "select s.s_name, s.s_address from supplier s, nation n "
        "where s.s_suppkey in ( "
        "  select ps.ps_suppkey from partsupp ps "
        "  where ps.ps_partkey in ( "
        "    select p.p_partkey from part p where p.p_name like 'antique%' ) "
        "  and ps.ps_availqty > ( "
        "    select 0.5 * sum(l.l_quantity) from lineitem l "
        "    where l.l_partkey = ps.ps_partkey and l.l_suppkey = ps.ps_suppkey "
        "    and l.l_shipdate >= date '1993-01-01' and l.l_shipdate < date '1994-01-01' )) "
        "and s.s_nationkey = n.n_nationkey and n.n_name = 'KENYA' order by s.s_name;",
    ),
    (
        "q22",
        "select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal from ( "
        "  select substring(c_phone from 1 for 2) as cntrycode, c_acctbal from customer c "
        "  where substring(c_phone from 1 for 2) in "
        "    ('24', '31', '11', '16', '21', '20', '34') "
        "  and c_acctbal > ( "
        "    select avg(c_acctbal) from customer "
        "    where c_acctbal > 0.00 and substring(c_phone from 1 for 2) in "
        "      ('24', '31', '11', '16', '21', '20', '34') ) "
        "  and not exists ( select * from orders o where o.o_custkey = c.c_custkey )) as custsale "
        "group by cntrycode order by cntrycode;",
    ),
]


def resolve_duckdb_db(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"DuckDB database not found: {path}")
        return path

    env = os.environ.get("SIRIUS_INTEGRATION_TEST_DB_PATH")
    if env and Path(env).is_file():
        return Path(env)

    for candidate in (
        REPO_ROOT / "test_datasets/tpch_sf1.duckdb",
        REPO_ROOT / "test/cpp/integration/data/duckdb/integration.duckdb",
    ):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "TPC-H DuckDB database not found. Set SIRIUS_INTEGRATION_TEST_DB_PATH "
        "or pass --duckdb-path."
    )


def build_session_sql(db_path: Path) -> str:
    escaped = str(db_path.resolve()).replace("'", "''")
    lines = [
        f"ATTACH IF NOT EXISTS '{escaped}' AS tpch (READ_ONLY);",
        "USE tpch;",
        ".timer on",
    ]
    for name, sql in QUERIES:
        lines.append(f".print __FOLD_QUERY_BEGIN__ {name}")
        lines.append(sql)
        lines.append(f".print __FOLD_QUERY_END__ {name}")
    lines.append(".print __FOLD_QUERY_END__ ALL_DONE")
    return "\n\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duckdb-path", help="Path to attached TPC-H DuckDB database")
    parser.add_argument(
        "--duckdb-bin", default=os.environ.get("SIRIUS_DUCKDB", str(DEFAULT_DUCKDB))
    )
    parser.add_argument(
        "--log-dir", type=Path, help="Directory for Sirius logs (SIRIUS_LOG_DIR)"
    )
    parser.add_argument(
        "--session-sql",
        type=Path,
        help="Write generated SQL here instead of a temp file",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print queries and exit without running"
    )
    parser.add_argument(
        "--no-pixi", action="store_true", help="Run duckdb directly (not via pixi run)"
    )
    args = parser.parse_args()

    print(f"Projection-fold candidates: {len(QUERIES)} queries")
    for name, sql in QUERIES:
        preview = " ".join(sql.split())
        if len(preview) > 90:
            preview = preview[:90] + "..."
        print(f"  {name}: {preview}")

    if args.dry_run:
        return 0

    try:
        db_path = resolve_duckdb_db(args.duckdb_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    session_sql = build_session_sql(db_path)

    if args.session_sql:
        args.session_sql.parent.mkdir(parents=True, exist_ok=True)
        sql_path = args.session_sql
    else:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sql", delete=False, prefix="fold_queries_"
        )
        tmp.write(session_sql)
        tmp.close()
        sql_path = Path(tmp.name)

    sql_path.write_text(session_sql, encoding="utf-8")
    print(f"\nDuckDB database: {db_path}")
    print(f"Session SQL:      {sql_path}")

    if not os.environ.get("SIRIUS_CONFIG_FILE") and DEFAULT_CONFIG.is_file():
        os.environ["SIRIUS_CONFIG_FILE"] = str(DEFAULT_CONFIG)

    env = os.environ.copy()
    env.setdefault("SIRIUS_LOG_LEVEL", "info")
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        env["SIRIUS_LOG_DIR"] = str(args.log_dir)
        print(f"Sirius logs:      {args.log_dir}")

    duckdb_bin = Path(args.duckdb_bin)
    if not duckdb_bin.is_file():
        print(
            f"ERROR: duckdb not found at {duckdb_bin}. Run: pixi run make",
            file=sys.stderr,
        )
        return 1

    cmd = [str(duckdb_bin), "-f", str(sql_path)]
    if not args.no_pixi:
        cmd = ["pixi", "run", *cmd]

    print(f"\nRunning: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
