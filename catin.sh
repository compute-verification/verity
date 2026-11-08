R=/workspace/research
echo "@@NEUTRAL"; cat $R/runs/$1/inputs/neutral.sh; echo "@@END"
echo "@@REQUEST"; python3 -c "
import json,sys;j=json.load(open('$R/requests/$1/request.json'));print(json.dumps({k:v for k,v in j.items() if k not in ('env',)})[:3000])"; echo "@@END"
ls $R/runs/$1/inputs
