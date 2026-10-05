"""TAHAP 5 - Model dokumen (MongoDB): satu dokumen 'Case 360' per case."""
from pymongo import ASCENDING, DESCENDING, MongoClient

from common import DATASET, MONGO_URI, OUTPUT_DIR, get_spark, lake_path, write_json


def to_doc(r):
    timeline = [{"event": "CREATED", "at": r["created_ts"], "via": r["channel"]},
                {"event": "ASSIGNED", "to": r["assigned_to"]}]
    if r["closed_ts"]:
        timeline.append({"event": "CLOSED", "at": r["closed_ts"]})
    return {
        "_id": int(r["case_number"]),
        "channel": r["channel"],
        "classification": {"category": r["category"], "subcategory": r["subcategory"],
                           "type": r["case_type"]},
        "assignment": {"team": r["assigned_to"]},
        "status": r["status"], "is_closed": bool(r["is_closed"]),
        "sla": {"status": r["sla_status"], "status_raw": r["sla_status_raw"],
                "within": None if r["is_within_sla"] is None else bool(r["is_within_sla"])},
        "created_at": r["created_ts"], "closed_at": r["closed_ts"],
        "metrics": {"resolution_hours": r["resolution_hours_clean"]},
        "timeline": timeline,
        "dq_flags": [f for f in (r["dq_flags"] or "").split(",") if f],
        "notes": [],  # di produksi: transkrip telepon, email, chat (data tidak terstruktur)
    }


def main():
    spark = get_spark("05_load_mongo")
    s = spark.read.parquet(lake_path("silver", DATASET))
    coll = MongoClient(MONGO_URI)["customer_care"]["cases"]
    coll.drop()
    batch, n = [], 0
    for row in s.toLocalIterator():
        batch.append(to_doc(row.asDict()))
        if len(batch) >= 2000:
            coll.insert_many(batch)
            n += len(batch)
            batch = []
    if batch:
        coll.insert_many(batch)
        n += len(batch)
    coll.create_index([("classification.category", ASCENDING), ("sla.within", ASCENDING)])
    coll.create_index([("created_at", DESCENDING)])
    coll.create_index([("assignment.team", ASCENDING)])

    pipeline = [
        {"$match": {"sla.within": {"$ne": None}}},
        {"$group": {"_id": "$classification.category", "cases": {"$sum": 1},
                    "within": {"$sum": {"$cond": ["$sla.within", 1, 0]}},
                    "avg_res_h": {"$avg": "$metrics.resolution_hours"}}},
        {"$project": {"cases": 1, "avg_res_h": {"$round": ["$avg_res_h", 2]},
                      "sla_compliance_pct": {"$round": [{"$multiply": [
                          {"$divide": ["$within", "$cases"]}, 100]}, 2]}}},
        {"$sort": {"cases": -1}},
    ]
    result = list(coll.aggregate(pipeline))
    write_json(OUTPUT_DIR / "mongo_aggregation_example.json",
               {"pipeline": pipeline, "result": result, "sample_document": coll.find_one()})
    print(f"[OK] {n:,} dokumen dimuat ke customer_care.cases")
    for r in result:
        print(f"     {r['_id']:30s} cases={r['cases']:>7,}  SLA={r['sla_compliance_pct']}%  avg={r['avg_res_h']} jam")


if __name__ == "__main__":
    main()
