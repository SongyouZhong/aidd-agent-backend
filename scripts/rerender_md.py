"""Re-render a broken MD report file from its JSON counterpart in S3."""
import asyncio
import json
import sys
import uuid

sys.path.insert(0, ".")


async def main(json_file_id_str: str, md_file_id_str: str, target_query: str) -> None:
    from app.db.engine import AsyncSessionLocal
    from app.models.session_file import SessionFile
    from app.services.report_renderer import render_target_report_md
    from app.storage.s3 import S3Storage

    json_file_id = uuid.UUID(json_file_id_str)
    md_file_id = uuid.UUID(md_file_id_str)

    s3 = S3Storage()
    await s3.start()
    try:
        async with AsyncSessionLocal() as db:
            json_row = await db.get(SessionFile, json_file_id)
            md_row = await db.get(SessionFile, md_file_id)
            if not json_row:
                print(f"JSON file {json_file_id} not found in DB")
                return
            if not md_row:
                print(f"MD file {md_file_id} not found in DB")
                return

            json_data = await s3.get_object(json_row.s3_key)
            if not json_data:
                print(f"JSON not found in S3: {json_row.s3_key}")
                return

            report = json.loads(json_data)
            print(f"target type: {type(report.get('target')).__name__}")
            print(f"target value: {repr(report.get('target'))[:100]}")

            md_payload = render_target_report_md(report, target_query).encode("utf-8")
            print(f"MD rendered OK, length: {len(md_payload)}")

            await s3.put_object(md_row.s3_key, md_payload, content_type="text/markdown")
            print(f"MD file updated in S3: {md_row.s3_key}")
    finally:
        await s3.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/rerender_md.py <json_file_id> <md_file_id> <target_query>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
