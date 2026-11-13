R="uv run -q --project tools/research research data sql"
$R "select kind, count(*) from artifacts group by kind order by 2 desc limit 30" 2>&1 | tail -32
echo "--- params.json paths in run records"
$R "select json_extract(a.manifest_json,'\$.meta.run_id') as run, json_extract(f.value,'\$.path') as path, json_extract(f.value,'\$.bytes') as bytes from artifacts a, json_each(a.manifest_json,'\$.payload.files') f where json_extract(f.value,'\$.path') like '%params.json' limit 200" 2>&1 | tail -202
echo "--- trimmed params.json"
$R "select json_extract(a.manifest_json,'\$.meta.run_id'), json_extract(f.value,'\$.path'), json_extract(f.value,'\$.bytes') from artifacts a, json_each(a.manifest_json,'\$.meta.trimmed') f where json_extract(f.value,'\$.path') like '%params.json' limit 100" 2>&1 | tail -102
