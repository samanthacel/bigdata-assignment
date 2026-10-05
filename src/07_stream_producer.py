"""(Opsional) Simulasi sumber streaming: replay case sebagai event JSON ke Kafka."""
import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd
from kafka import KafkaProducer

from common import KAFKA_BOOTSTRAP, KAFKA_TOPIC, latest_manifest, norm_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=20, help="event per detik (0 = secepatnya)")
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()
    df = pd.read_csv(latest_manifest()["local_csv"], dtype=str, keep_default_na=False)
    df.columns = [norm_col(c) for c in df.columns]
    prod = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP, linger_ms=20,
                         key_serializer=lambda k: str(k).encode(),
                         value_serializer=lambda v: json.dumps(v).encode())
    for i, rec in enumerate(df.head(a.limit).to_dict("records"), 1):
        rec["event_time"] = datetime.now(timezone.utc).isoformat()
        prod.send(KAFKA_TOPIC, key=rec.get("case_number"), value=rec)
        if i % 500 == 0:
            print(f"{i:,} event terkirim")
        if a.rate > 0:
            time.sleep(1.0 / a.rate)
    prod.flush()
    print("[OK] selesai mengirim event")


if __name__ == "__main__":
    main()
