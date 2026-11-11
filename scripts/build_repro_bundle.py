"""Build a small local upload bundle. Never uploads anything."""
import argparse
import json
from pathlib import Path
import zipfile
from scripts.repro_common import ROOT,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/stage4-inputs.zip')
    args=parser.parse_args()
    names=['scripts/__init__.py','scripts/repro_common.py','scripts/repro_workflow.py',
           'scripts/repro_evaluate.py','scripts/repro_colab_setup.py','scripts/compare_repro.py',
           'configs/repro-a.yaml','configs/repro-b.yaml','configs/repro-experiment.json',
           'configs/repro-policy.json','configs/repro-requirements.in','configs/repro-requirements.lock.txt',
           'fixtures/repro-train.jsonl','fixtures/repro-eval.jsonl',
           'notebooks/stage-4-reproducibility.ipynb','docs/test-results.md']
    manifest={'purpose':'Stage 4 inputs only; no training results',
              'sha256':{name:sha256(ROOT/name) for name in names}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT/name,name)
        archive.writestr('bundle-manifest.json',json.dumps(manifest,indent=2)+'\n')
    print(args.output)
    print('sha256:',sha256(args.output),'bytes:',args.output.stat().st_size)


if __name__=='__main__':
    main()
