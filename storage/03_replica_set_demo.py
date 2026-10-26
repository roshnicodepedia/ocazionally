"""
Unit 3 (NoSQL), step 3: replication, failover and the CAP theorem - live.

Run from the project root (venv active, MongoDB main service running):
    python storage/03_replica_set_demo.py

What it does, with real MongoDB servers on your laptop:
  1. starts THREE mongod servers (ports 27018-27020) forming one replica set
  2. copies the vendor collection into it and checks all 3 copies match
  3. KILLS the primary and times how long another server takes to take over
  4. writes again (it works: 2 of 3 servers are enough = a majority)
  5. brings the dead server back and shows it catching up
  6. KILLS TWO servers: the survivor refuses writes (CAP: consistency over availability)

It leaves your normal MongoDB (port 27017) and your data untouched.
A Windows Firewall box may appear once: click Allow.
Everything printed is also saved to benchmarks/replica_set_demo_log.txt.
"""
import argparse
import glob
import os
import shutil
import socket
import subprocess
import sys
import time

import psutil
from pymongo import MongoClient, WriteConcern
from pymongo.errors import PyMongoError

RS_NAME = "ocazionally-rs"
PORTS = [27018, 27019, 27020]
HOST = "127.0.0.1"
ROOT = os.path.join("storage", "rs_data")
LOG_PATH = os.path.join("benchmarks", "replica_set_demo_log.txt")

ap = argparse.ArgumentParser()
ap.add_argument("--mongod", default=None, help="full path to mongod.exe")
args = ap.parse_args()

os.makedirs("benchmarks", exist_ok=True)
_log = open(LOG_PATH, "w", encoding="utf-8")
T0 = time.time()


def say(msg="", stamp=True):
    line = f"[{time.time() - T0:5.1f}s] {msg}" if (stamp and msg) else msg
    print(line, flush=True)
    _log.write(line + "\n")
    _log.flush()


def find_mongod():
    if args.mongod:
        return args.mongod
    hits = glob.glob(r"C:\Program Files\MongoDB\Server\*\bin\mongod.exe")
    if hits:
        return sorted(hits)[-1]
    found = shutil.which("mongod")
    if found:
        return found
    sys.exit("Could not find mongod. Run again with --mongod \"C:\\path\\to\\mongod.exe\"")


MONGOD = find_mongod()
procs = {}                                   # port -> Popen


def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, port)) == 0


def kill_leftovers():
    """Stop any mongod from an earlier run of this demo (never the main service)."""
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "mongod" in (p.info["name"] or "").lower() and RS_NAME in cmd:
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(1.5)


def start_node(port, fresh=False):
    path = os.path.join(ROOT, f"node{port}")
    if fresh and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    cmd = [MONGOD, "--replSet", RS_NAME, "--port", str(port), "--bind_ip", HOST,
           "--dbpath", path, "--logpath", os.path.join(path, "mongod.log"),
           "--oplogSize", "128", "--wiredTigerCacheSizeGB", "0.25"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    procs[port] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=flags)


def direct(port, secondary_ok=False, timeout=2000):
    extra = "&readPreference=secondaryPreferred" if secondary_ok else ""
    return MongoClient(f"mongodb://{HOST}:{port}/?directConnection=true{extra}",
                       serverSelectionTimeoutMS=timeout)


def ping(port):
    with direct(port) as c:
        return c.admin.command("ping")


def wait_until(fn, what, timeout=90, every=1.0):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        try:
            r = fn()
            if r:
                return r
        except PyMongoError as e:
            last = e
        time.sleep(every)
    raise RuntimeError(f"Timed out waiting for {what}. Last error: {last}")


def role(port):
    """'PRIMARY', 'SECONDARY' or None (down)."""
    try:
        with direct(port, timeout=1500) as c:
            h = c.admin.command("hello")
        if h.get("isWritablePrimary"):
            return "PRIMARY"
        if h.get("secondary"):
            return "SECONDARY"
        return "STARTING"
    except PyMongoError:
        return None


def find_primary(candidates=None):
    for port in candidates or PORTS:
        if role(port) == "PRIMARY":
            return port
    return None


def count_on(port):
    with direct(port, secondary_ok=True) as c:
        return c["ocazionally"]["vendors"].count_documents({})


def show_members(title):
    say(title)
    for port in PORTS:
        r = role(port) or "DOWN"
        n = "-"
        if r in ("PRIMARY", "SECONDARY"):
            try:
                n = f"{count_on(port):,}"
            except PyMongoError:
                n = "?"
        say(f"      port {port}   {r:<10} vendors stored: {n}", stamp=False)


def rs_client(timeout=10000):
    hosts = ",".join(f"{HOST}:{p}" for p in PORTS)
    return MongoClient(f"mongodb://{hosts}/?replicaSet={RS_NAME}", serverSelectionTimeoutMS=timeout)


def stop_all():
    for port, p in list(procs.items()):
        if p.poll() is None:
            p.kill()
    kill_leftovers()


try:
    say("=" * 70, stamp=False)
    say("REPLICATION AND FAILOVER DEMO (3-server MongoDB replica set)", stamp=False)
    say("=" * 70, stamp=False)
    say(f"Using {MONGOD}")

    # ---- 0. clean start ------------------------------------------------------------------
    kill_leftovers()
    busy = [p for p in PORTS if port_in_use(p)]
    if busy:
        sys.exit(f"Ports {busy} are in use by another program. Close it and try again.")
    shutil.rmtree(ROOT, ignore_errors=True)

    # ---- 1. start three servers ----------------------------------------------------------
    say("STEP 1  Starting three MongoDB servers ...")
    for port in PORTS:
        start_node(port, fresh=True)
    for port in PORTS:
        wait_until(lambda p=port: ping(p), f"server {port} to start")
    say("        all three servers are up")

    cfg = {"_id": RS_NAME,
           "settings": {"electionTimeoutMillis": 5000},       # faster failover for the demo
           "members": [{"_id": i, "host": f"{HOST}:{p}"} for i, p in enumerate(PORTS)]}
    with direct(PORTS[0], timeout=5000) as c:
        c.admin.command("replSetInitiate", cfg)
    say("        replica set created, waiting for the first election ...")
    primary = wait_until(find_primary, "a primary to be elected")
    say(f"        server {primary} was elected PRIMARY")
    wait_until(lambda: all(role(p) in ("PRIMARY", "SECONDARY") for p in PORTS), "all members ready")

    # ---- 2. copy data and verify replication -------------------------------------------------
    say("STEP 2  Copying the vendors collection into the replica set ...")
    source = MongoClient(f"mongodb://{HOST}:27017", serverSelectionTimeoutMS=3000)
    docs = list(source["ocazionally"]["vendors"].find())
    source.close()
    if not docs:
        sys.exit("The main MongoDB has no vendors. Run storage/01_load_mongo.py first.")
    client = rs_client()
    vendors = client["ocazionally"].get_collection("vendors", write_concern=WriteConcern(w="majority"))
    vendors.insert_many(docs)
    say(f"        wrote {len(docs):,} vendor documents (acknowledged by a majority)")
    wait_until(lambda: all(count_on(p) == len(docs) for p in PORTS), "copies to match", timeout=60)
    show_members("        every server now holds a full copy:")

    # ---- 3. kill the primary -----------------------------------------------------------------
    primary = find_primary()
    say(f"STEP 3  Simulating a crash: killing the PRIMARY (port {primary}) ...")
    t_kill = time.time()
    procs[primary].kill()
    survivors = [p for p in PORTS if p != primary]
    new_primary = wait_until(lambda: find_primary(survivors), "a new primary", timeout=60, every=0.5)
    election_s = time.time() - t_kill
    say(f"        server {new_primary} took over as PRIMARY after {election_s:.1f} seconds")

    # ---- 4. keep writing ------------------------------------------------------------------------
    say("STEP 4  Writing while one server is down ...")
    extra = {"_id": "V-FAILOVER-TEST", "name": "Written during the outage", "category": "other",
             "location": {"city": "Ahmedabad", "area": None}, "source": "demo"}

    def write_once():
        client["ocazionally"].get_collection("vendors", write_concern=WriteConcern(w="majority")
                                             ).replace_one({"_id": extra["_id"]}, extra, upsert=True)
        return True
    wait_until(write_once, "a successful write", timeout=30, every=1.0)
    say("        write ACCEPTED (2 of 3 servers form a majority, so the system stays available)")

    # ---- 5. bring the old primary back ------------------------------------------------------
    say(f"STEP 5  Restarting the crashed server (port {primary}) ...")
    start_node(primary)
    wait_until(lambda: role(primary) == "SECONDARY", "the old primary to rejoin", timeout=90)
    wait_until(lambda: count_on(primary) == len(docs) + 1, "it to catch up", timeout=60)
    say("        it rejoined as a SECONDARY and caught up, including the write it missed")
    show_members("        cluster state:")

    # ---- 6. CAP: lose the majority ------------------------------------------------------------
    primary_now = find_primary()
    others = [p for p in PORTS if p != primary_now]
    victim = others[0]
    survivor = others[1]
    say(f"STEP 6  CAP theorem: killing the primary ({primary_now}) AND a secondary ({victim}) ...")
    procs[primary_now].kill()
    procs[victim].kill()
    time.sleep(8)
    say(f"        only server {survivor} is left, so no majority is possible")
    try:
        lone = rs_client(timeout=7000)
        lone["ocazionally"].get_collection("vendors", write_concern=WriteConcern(w="majority")
                                           ).insert_one({"_id": "V-SHOULD-FAIL", "name": "no majority"})
        say("        (unexpected) the write was accepted")
    except PyMongoError as e:
        say(f"        write REFUSED: {type(e).__name__}")
        say("        -> the cluster chose CONSISTENCY over AVAILABILITY (it will not accept a write", stamp=False)
        say("           it cannot safely replicate)", stamp=False)
    try:
        n = count_on(survivor)
        say(f"        reads still work on the lone server (may be slightly stale): {n:,} vendors")
    except PyMongoError as e:
        say(f"        reads failed too: {type(e).__name__}")

    say("")
    say("SUMMARY FOR YOUR REPORT", stamp=False)
    say("  - 3-member replica set, every vendor stored 3 times (replication factor 3)", stamp=False)
    say(f"  - primary crash -> automatic election -> new primary in {election_s:.1f}s", stamp=False)
    say("  - writes continue with 2 of 3 servers (majority write concern)", stamp=False)
    say("  - the returning server re-synced without manual work", stamp=False)
    say("  - with only 1 of 3 left, writes are refused: consistency over availability (CP)", stamp=False)

finally:
    say("")
    say("Cleaning up: stopping the three demo servers ...")
    try:
        stop_all()
    finally:
        say("Done. Log saved to " + LOG_PATH)
        _log.close()