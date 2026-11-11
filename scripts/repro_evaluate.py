"""Identical CPU/float32 held-out evaluation of the two REAL saved adapters."""
import argparse
import hashlib
import json
import math

from scripts.repro_common import ROOT,COLAB_LOGS,read_json,write_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['a','b'],required=True)
    arm=parser.parse_args().arm
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import PeftModel
    spec=read_json(ROOT/'configs/repro-experiment.json')
    receipt=read_json(COLAB_LOGS/f'run-{arm}/receipt.json')
    if not receipt['complete_artifacts']:
        raise RuntimeError('No complete real training artifacts')
    directory=ROOT/receipt['artifact_directory']
    if (directory/'evaluation.json').exists():
        raise FileExistsError('Evaluation already exists; do not overwrite')
    settings=spec['evaluation']
    torch.set_num_threads(settings['threads'])
    torch.use_deterministic_algorithms(True)
    set_seed(settings['seed'])
    base=ROOT/spec['model_directory']
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
                dtype=torch.float32,attn_implementation='eager')
    model=PeftModel.from_pretrained(model,directory,local_files_only=True,is_trainable=False)
    model.eval()
    records=[]
    with torch.inference_mode():
        for line in (ROOT/'fixtures/repro-eval.jsonl').read_text().splitlines():
            item=json.loads(line)
            prompt=tokenizer(item['prompt'],add_special_tokens=True)['input_ids']
            target=tokenizer(item['target'],add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
            tokens=torch.tensor([prompt+target],dtype=torch.long)
            labels=torch.tensor([[-100]*len(prompt)+target],dtype=torch.long)
            nll=float(model(input_ids=tokens,attention_mask=torch.ones_like(tokens),labels=labels).loss)
            inputs=torch.tensor([prompt],dtype=torch.long)
            generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                do_sample=False,max_new_tokens=settings['max_new_tokens'],
                pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
            new_ids=generated[0,len(prompt):].tolist()
            text=tokenizer.decode(new_ids,skip_special_tokens=True)
            if not math.isfinite(nll):
                raise ValueError('Non-finite measured evaluation loss')
            records.append({**item,'prompt_ids':prompt,'target_ids':target,'target_tokens':len(target),
                'nll':nll,'generated_ids':new_ids,'generated_text':text,
                'exact_match':text.strip()==item['target']})
    mean=sum(r['nll']*r['target_tokens'] for r in records)/sum(r['target_tokens'] for r in records)
    identity={'settings':settings,'model_revision':spec['model_revision'],
              'eval_data_sha256':sha256(ROOT/'fixtures/repro-eval.jsonl'),
              'tokenized_inputs_sha256':hashlib.sha256(json.dumps(
                  [(r['prompt_ids'],r['target_ids']) for r in records],sort_keys=True).encode()).hexdigest()}
    result={'identity':identity,'adapter_sha256':sha256(ROOT/receipt['adapter_path']),
            'examples':records,'mean_nll':mean,
            'exact_match_accuracy':sum(r['exact_match'] for r in records)/len(records),
            'note':'Independent deterministic CPU reload evaluation, not Soup training loss or a general quality benchmark.'}
    write_json(directory/'evaluation.json',result)
    print(json.dumps({'arm':arm,'examples':len(records),'mean_nll':mean,
                      'exact_match_accuracy':result['exact_match_accuracy']},indent=2))


if __name__=='__main__':
    main()
