import json


vals_gt= {'dress':'/data/oss_bucket_0/lida/cir/data/fashionIQ/captions/cap.dress.val.json',
'shirt':'/data/oss_bucket_0/lida/cir/data/fashionIQ/captions/cap.shirt.val.json',
'toptee':'/data/oss_bucket_0/lida/cir/data/fashionIQ/captions/cap.toptee.val.json'}


vals_result= {'dress':'/data/oss_bucket_0/lida/cir/store/dress/result.json',
'shirt':'/data/oss_bucket_0/lida/cir/data/fashionIQ/captions/cap.shirt.val.json',
'toptee':'/data/oss_bucket_0/lida/cir/data/fashionIQ/captions/cap.toptee.val.json'}


def recall(res_list: list[list[str]], label_list: list[list[str]], k: int) -> float:
        true_positives = 0
        false_negatives = 0
        for actual, predicted in zip(label_list, res_list):
            actual_set = set(actual)
            predicted_set = set(predicted[:k])
            true_positives += len(actual_set & predicted_set)
            false_negatives += len(actual_set - predicted_set)
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0


types = ['dress']

for single_type in types:
    with open(vals_gt[single_type],mode='r',encoding='utf-8') as rf:
        gt_data = json.load(rf)
    gt_labels = {}
    for single_gt in gt_data:
        gt_labels[single_gt['candidate']] = single_gt['target']
    
    with open(vals_result[single_type],mode='r',encoding='utf-8') as rf:
        retrieval_results = json.load(rf)
    

    total_results = [] 

    for single_result in retrieval_results:
        ref_image = single_result['image_name']
        ref_reasoning = single_result['generated_text'].replace('<|im_end|>','')
        recall_result_list = single_result['result']
        format_results = []
        for single_recall_result in recall_result_list:
            format_single_info = {
                # 'reasoning':single_recall_result['meta_info']['generated_text'].replace('<|im_end|>',''),
            'image_name':single_recall_result['meta_info']['image_name'],
            'ranking':single_recall_result['rank'],
            'dis':single_recall_result['distance'],}
            format_results.append(format_single_info)
        temp = {'ref_image':ref_image,
        # 'ref_reasoning':ref_reasoning,
        'target':gt_labels[ref_image],
        'result':format_results}
        # print(temp)
        total_results.append(temp)
        # total_results.append({'ref_image':ref_image,'ref_reasoning':ref_reasoning,'target':gt_labels[ref_image],
        # 'result':format_results})

    
    cutoffs=[5,10,20]


    labels_gt = [] 
    preds = [] 

    for single_metric in total_results:
        single_label = single_metric['target']
        print(single_metric['ref_image'])
        print(single_label)
        # print(single_label)
        single_pred = [item['image_name'] for item in single_metric['result']]
        print(single_metric['result'])
        
        # break
        # print(single_pred)
        print('_______')
        labels_gt.append([single_label])
        preds.append(single_pred)
    #     labels = [[item['target']] for item in total_results]
    #     # print(total_results[0])
    #     retrievals = []
    #     for single_result in retrieval_results:
    #         print(single_result)
    #         result = [item['image_name'] for item in single_result['result']]
    #         retruevals.append(result)
    #     # retrievals = [item['image_name'] for single_result in total_results for item in single_result['result']]
    #     print(retrievals[0])
    #     # retrievals = [item[:cut] for item in retrievals]

    for cut in cutoffs:
        print(f'Recall@{cut}:', recall(labels_gt,preds,cut))