R="uv run -q --project tools/research research data sql"
echo "--- any artifact whose manifest names a params.json, by kind"
$R "select kind, count(*) from artifacts where manifest_json like '%params.json%' group by kind" 2>&1 | tail -20
echo "--- non run-record paths"
$R "select a.kind, a.id, json_extract(f.value,'\$.path'), json_extract(f.value,'\$.bytes') from artifacts a, json_each(a.manifest_json,'\$.payload.files') f where a.kind != 'run-record/v1' and json_extract(f.value,'\$.path') like '%params.json' limit 60" 2>&1 | tail -62
echo "--- the dry run's job"
$R "select substr(manifest_json, 1, 50) , json_extract(f.value,'\$.path') from artifacts a, json_each(a.manifest_json,'\$.payload.files') f where json_extract(a.manifest_json,'\$.meta.run_id')='r20261104-010007-0433' and json_extract(f.value,'\$.path') not like 'out/dry/%/private/%' limit 60" 2>&1 | tail -62
uv run -q --project tools/research research data get-run r20261104-010007-0433 --help 2>&1 | head -3
