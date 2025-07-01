import json


brief_description_file = '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_tgt_200_GT.json'
complex_instruction_file = '/mnt/ssd/liuchao/PointLLM/PointLLM_complex_instruction_70K.json'
output_file = '/mnt/ssd/liuchao/PointLLM/PointLLM_detailed_description_tgt_200_GT.json'

with open(brief_description_file, 'r') as f:
    brief_descriptions = json.load(f)
    
with open(complex_instruction_file, 'r') as f:
    complex_instructions = json.load(f)
    
print(f"Loaded {len(brief_descriptions)} brief descriptions and {len(complex_instructions)} complex instructions.")
   
statistic_object_ids = set()
for brief_description in brief_descriptions:
    object_id = brief_description['object_id']
    statistic_object_ids.add(object_id)
    # brief_description_text = brief_description['brief_description']
    
    # # Find the corresponding complex instruction
    # for complex_instruction in complex_instructions:
    #     if complex_instruction['object_id'] == object_id:
    #         complex_instruction['conversations'].insert(0, brief_description_text)
    #         break
    

detsailed_descriptions = []
for complex_instruction in complex_instructions:
    object_id = complex_instruction['object_id']
    if object_id not in statistic_object_ids:
        continue
    if complex_instruction["conversation_type"] != 'detailed_description':
        continue
    
    detsailed_description = complex_instruction.copy()
    detsailed_description.remove('conversation_type')
    detsailed_descriptions.append(detsailed_description)
    
with open(output_file, 'w') as f:
    json.dump(detsailed_descriptions, f, indent=4)