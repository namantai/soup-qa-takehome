"""CPU-only real-tokenizer probe of installed Soup formatting and label masks.

No model/torch loading. Independent expectations use character spans in the
explicit ChatML rendering and fast-tokenizer offsets, not Soup mask helpers.
"""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path

import yaml
from scripts.repro_common import ROOT,write_json,sha256


def manual_chatml(messages):
    text=''; spans=[]
    for index,message in enumerate(messages):
        text+='<|im_start|>'+message['role']+'\n'
        start=len(text);text+=message['content']
        spans.append((index,message['role'],start,len(text)))
        text+='<|im_end|>\n'
    return text,spans


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from transformers import AutoTokenizer
    from soup_cli.config.schema import SoupConfig
    from soup_cli.data.sft_format import build_format_row
    from soup_cli.data.loss_mask import ensure_causal_loss_target
    config=yaml.safe_load((ROOT/'configs/repro-a.yaml').read_text())
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    assert importlib.metadata.version('transformers')=='5.17.0'
    rows=[json.loads(line) for line in (ROOT/'fixtures/repro-train.jsonl').read_text().splitlines()]
    first=rows[0]
    messages=[{'role':'user','content':first['instruction']+'\n'+first['input']},
              {'role':'assistant','content':first['output']}]
    raw_mask=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,
                    return_assistant_tokens_mask=True,return_dict=True)
    results=[];problems=[]
    def inspect(label,cfg,items,expectation):
        parsed=SoupConfig.model_validate(cfg)
        formatter=build_format_row(tokenizer,parsed.data,training_cfg=parsed.training)
        rendered,spans=manual_chatml(items)
        assert tokenizer.apply_chat_template(items,tokenize=False,add_generation_prompt=False)==rendered
        offsets=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        output=formatter({'messages':items})
        # Save actual inputs/arrays even if a following audit assertion fails.
        write_json(args.output.parent/'cases'/f'{label}.json',
                   {'config':cfg,'messages':items,'actual':output,'rendered':rendered,
                    'offset_tokenization':dict(offsets)})
        reference=offsets['input_ids'][:parsed.data.max_length]
        positions=list(offsets['offset_mapping'][:parsed.data.max_length])
        if expectation=='full' and output['input_ids']==reference+[tokenizer.eos_token_id]:
            # Language-model training may append one EOS. It is a separate
            # format token; do not misclassify it as prompt/answer content.
            reference=reference+[tokenizer.eos_token_id]
            positions.append((len(rendered),len(rendered)))
        assert output['input_ids']==reference
        tokens=[]
        for i,(token,truth,span) in enumerate(zip(output['input_ids'],output['labels'],positions)):
            start,end=span
            content=[(index,role) for index,role,a,b in spans if start<b and end>a]
            role=content[0][1] if content else 'format'
            index=content[0][0] if content else None
            tokens.append({'i':i,'id':token,'text':tokenizer.decode([token]),'offset':list(span),
                           'content_role':role,'message_index':index,'label':truth,'supervised':truth!=-100})
        issues=[]
        if not any(t['supervised'] for t in tokens[1:]):issues.append('no causal targets')
        if expectation=='assistant':
            if any(t['supervised'] for t in tokens if t['content_role']=='user'):issues.append('user content trained')
            if not all(t['supervised'] for t in tokens if t['content_role']=='assistant'):issues.append('assistant content masked')
        elif expectation=='full':
            if not all(t['supervised'] for t in tokens):issues.append('full-sequence mode ignored')
        elif expectation=='last':
            final=max(i for i,r,a,b in spans if r=='assistant')
            if any(t['supervised'] for t in tokens if t['content_role']=='user' or (t['content_role']=='assistant' and t['message_index']!=final)):issues.append('history/prompt still trained')
            if not all(t['supervised'] for t in tokens if t['message_index']==final):issues.append('last answer masked')
        item={'label':label,'config':cfg,'messages':items,'rendered':rendered,'input_ids':output['input_ids'],
              'labels':output['labels'],'attention_mask':output['attention_mask'],'tokens':tokens,
              'expected':expectation,'problems':issues,
              'supervised_tokens':sum(t['supervised'] for t in tokens),
              'supervised_format_tokens':sum(t['supervised'] and t['content_role']=='format' for t in tokens)}
        results.append(item);problems.extend(f'{label}: {x}' for x in issues)
        return output
    original=inspect('stage4-first-row',config,messages,'assistant')
    full=deepcopy(config);full['data'].update(train_on_responses_only=False,train_on_prompt=True)
    changed=inspect('train-on-prompt',full,messages,'full')
    assert original['labels']!=changed['labels'], 'Configuration toggle had no effect'
    multi=messages+[{'role':'user','content':'Convert river to uppercase.'},{'role':'assistant','content':'RIVER'}]
    inspect('history-default',config,multi,'assistant')
    history=deepcopy(config);history['data']['mask_history']=True
    inspect('history-last-only',history,multi,'last')
    # Check every actual Stage 4 row, retaining real ids/labels and char-span audit.
    for index,row in enumerate(rows):
        items=[{'role':'user','content':row['instruction']+'\n'+row['input']},
               {'role':'assistant','content':row['output']}]
        inspect(f'dataset-{index:02d}',config,items,'assistant')
    truncated=deepcopy(config);truncated['data']['max_length']=64
    truncated_messages=[{'role':'user','content':'word '*128},
                        {'role':'assistant','content':'APPLE'}]
    write_json(args.output.parent/'truncation-input.json',
               {'config':truncated,'messages':truncated_messages})
    parsed=SoupConfig.model_validate(truncated)
    truncated_output=build_format_row(tokenizer,parsed.data,training_cfg=parsed.training)({'messages':truncated_messages})
    write_json(args.output.parent/'truncation-output.json',truncated_output)
    try:
        # The formatter truncates; live _map_text_sft_rows then calls this
        # separate guard. Test the guard instead of attributing it to formatter.
        ensure_causal_loss_target(truncated_output['labels'],max_length=parsed.data.max_length)
    except ValueError as exc:
        truncation={'exception':type(exc).__name__,'message':str(exc),'expected':'reject row with no remaining answer targets'}
    else:
        truncation={'exception':None};problems.append('answer-free truncation accepted')
    report={'kind':'Actual CPU tokenizer/Soup formatter execution, no training',
        'versions':{n:importlib.metadata.version(n) for n in ['soup-cli','transformers','tokenizers','jinja2']},
        'tokenizer_sha256':{p.name:sha256(p) for p in args.tokenizer.iterdir() if p.is_file() and p.name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']},
        'raw_hf_assistant_mask':list(raw_mask['assistant_masks']),
        'raw_mask_all_zero':not any(raw_mask['assistant_masks']),
        'template_has_generation_marker':'{% generation' in tokenizer.chat_template,
        'cases':results,'truncation':truncation,'problems':problems,
        'limitations':'No training rerun. Actual tokenizer and installed formatter; no custom tokenizer. Legacy ChatML fallback may supervise role/delimiter tokens as well as content.'}
    write_json(args.output,report)
    print(json.dumps({'raw_mask_all_zero':report['raw_mask_all_zero'],'cases':len(results),
       'first_row_supervised':results[0]['supervised_tokens'],'first_row_format_supervised':results[0]['supervised_format_tokens'],
       'truncation':truncation,'problems':problems},indent=2))
    return int(bool(problems))


if __name__=='__main__':
    raise SystemExit(main())
