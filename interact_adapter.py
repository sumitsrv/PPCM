from tabulate import tabulate
tabulate.PRESERVE_WHITESPACE = True
from utils.helper import load_classifier
from utils.helper import EOS_ID
from utils.utils_sample import scorer
import torch.nn.functional as F
import torch
from nltk import tokenize
from models.wd import weight_decoder
from utils.helper import cut_seq_to_eos

#CUDA_VISIBLE_DEVICES=2 python main.py -D sentiment --label_class 3 --length 30 --num_samples 1 --interact --verbose --speaker DGPT --load_check_point_adapter runs/SENT_very_negative_Mar30_13-59-53/pytorch_model.bin

def top_k_logits(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def sample(model, args, context=None, past=None, device='cuda',
                       sample=True, repetition_penalty=1.0, weighed_tokens=None):
    output = torch.tensor(context, device=device, dtype=torch.long) if context else None 
    # Creates tensor of the size of context. The length of Context is just the num_samples passed from the command line. num_samples denote the number of different candidate sentences we would like to generate and rate.
    output_response = output.new_zeros([output.size(0),0])
    stopped = [0 for _ in range(output.size(0))]
    for i in range(args.length):

        if past is None and output is not None:
            prev = output[:, -1:]
            _, past = model(output[:, :-1])

        # print("Sample idx shape:", prev.shape, "Past:", len(past))
        logits, past = model(prev, past=past) # Get the predictions from the last layer using the previously generated token (prev) and the past tokens (excluding the first token in the previous series on input).

        # print("Logits: ", len(logits))
        logits = logits[:, -1, :] / args.temperature  # + SmallConst
        for i_o, o_ in enumerate(output):
            for token_idx in set(o_.tolist()):
                if logits[i_o, token_idx] < 0:
                    logits[i_o, token_idx] *= repetition_penalty
                else:
                    logits[i_o, token_idx] /= repetition_penalty

        # print("Logits: ", len(logits))
        
        logits = top_k_logits(logits, k=args.top_k)  # + SmallConst
        
        # Modify log_probs to enhance the weight of the tokens received from the user.

        if weighed_tokens:
            weighed_tokens_tensor = torch.zeros(logits.size(dim=1))
            print("WTT: ", weighed_tokens_tensor.shape)
            [_ , vocab_size] = logits.shape
            one_hot_vectors = []
            # for good_list in weighed_tokens:
                # good_list = list(filter(lambda x: len(x) <= 1, good_list))
            # weighed_tokens_tensor = torch.tensor(weighed_tokens_tensor)
            # num_good = weighed_tokens_tensor.shape[0]
            one_hot_good = torch.zeros(1, vocab_size)
            print("OHG: ", one_hot_good.shape)
            weighed_tokens = torch.transpose(torch.tensor(weighed_tokens), 0, 1)
            print("WTS: ", weighed_tokens)
            
            one_hot_good.scatter_(1, weighed_tokens, one_hot_good)
            # one_hot_good = torch.sum(one_hot_good, dim=0)
            # one_hot_vectors.append(one_hot_good)

            # log_probs = log_probs + args.bow_scale_weight*one_hot_vectors[-1]*log_probs #+ args.bow_scale_weight*one_hot_vectors[-1]
            
            # for token in weighed_tokens:
            #     logits[token] *= 10

        log_probs = F.softmax(logits, dim=-1)

        if sample:
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            _, prev = torch.topk(log_probs, k=1, dim=-1)

        # print("Logits: ", logits.shape, "Log probs: ", log_probs.shape, "Prev: ", prev.shape)
        output = prev if output is None else torch.cat((output, prev), dim=1)  # update output

        # print("Output:", len(output))
        output_response = torch.cat((output_response, prev), dim=1)
        # print("Output response:", output_response)
        for i_p, p in enumerate(prev.tolist()):
            if(p[0]) == EOS_ID:
                stopped[i_p] = 1

        if(all(x == 1 for x in stopped)): break

    return output_response

def get_rankers(args,model):
    classifiers = {}

    args.discrim = 'sentiment'
    args.label_class = 2
    classifier, class2idx = load_classifier(args, model)
    classifiers['a'] = [classifier, class2idx]

    args.discrim = 'sentiment'
    args.label_class = 3
    classifier, class2idx = load_classifier(args, model)
    classifiers['b'] = [classifier, class2idx]

    args.discrim = 'daily_dialogue_act'
    args.label_class = 1
    classifier, class2idx = load_classifier(args, model)
    classifiers['c'] = [classifier, class2idx]

    args.discrim = 'toxicity'
    args.label_class = 1
    classifier, class2idx = load_classifier(args, model)
    classifiers['d'] = [classifier, class2idx]
    
    args.discrim = 'AG_NEWS'
    args.label_class = 0
    classifier, class2idx = load_classifier(args, model)
    classifiers['e'] = [classifier, class2idx]

    args.discrim = 'AG_NEWS'
    args.label_class = 1
    classifier, class2idx = load_classifier(args, model)
    classifiers['f'] = [classifier, class2idx]

    args.discrim = 'AG_NEWS'
    args.label_class = 2
    classifier, class2idx = load_classifier(args, model)
    classifiers['g'] = [classifier, class2idx]

    args.discrim = 'AG_NEWS'
    args.label_class = 3
    classifier, class2idx = load_classifier(args, model)
    classifiers['h'] = [classifier, class2idx]

    return classifiers

def interact(args,model,enc,classifier,class2idx,device):
    classifiers = get_rankers(args,model)
    history = []
    while True:
        raw_text = input("USR >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("USR >>>")
        
        if len(history) == 0:
            idx_target_utterance_for_alignment = 0
        else:
            idx_target_utterance_for_alignment = input(f"TRGT >>>> Between 0 and {len(history)}")

            while not idx_target_utterance_for_alignment or idx_target_utterance_for_alignment not in range(len(history)):
                print(f'Target utterance should be between 0 and {len(history)}!')
                idx_target_utterance_for_alignment = input(f"TRGT >>>> Between 0 and {len(history)}")
    

        classifier,class2idx = classifiers["b"]
        # args.num_samples = 10
        task_id = 0 
        args.label_class = 3
    
        # style = input("Choose a style \n (a) Positive (b) Negative (c) Question (d) Toxic (e) World (f) Sports (g) Business (h) Sci/Tech (i) DGPT \n >>> ")
        # if(style == "a"): 
        #     classifier,class2idx = classifiers["a"]
        #     # args.num_samples = 10
        #     task_id = 1 
        #     args.label_class = 2
        # elif(style == "b"): 
        #     classifier,class2idx = classifiers["b"]
        #     # args.num_samples = 10
        #     task_id = 0 
        #     args.label_class = 3
        # elif(style == "c"): 
        #     classifier,class2idx = classifiers["c"]
        #     # args.num_samples = 10
        #     task_id = 3
        #     args.label_class = 1
        # elif(style == "d"): 
        #     classifier,class2idx = classifiers["d"]
        #     # args.num_samples = 10
        #     task_id = 2 
        #     args.label_class = 1
        # elif(style == "e"): 
        #     classifier,class2idx = classifiers["e"]
        #     # args.num_samples = 10
        #     task_id = 7 
        #     args.label_class = 0
        # elif(style == "f"): 
        #     classifier,class2idx = classifiers["f"]
        #     # args.num_samples = 10
        #     task_id = 6 
        #     args.label_class = 1

        # elif(style == "g"): 
        #     classifier,class2idx = classifiers["g"]
        #     # args.num_samples = 10
        #     task_id = 4 
        #     args.label_class = 2

        # elif(style == "h"): 
        #     classifier,class2idx = classifiers["h"]
        #     # args.num_samples = 10
        #     task_id = 5 
        #     args.label_class = 3
        # else:
        #     # args.num_samples = 1
        #     args.label_class = 0
        #     task_id = -1


        history.append(raw_text)
        target_utterance_for_alignment = history[idx_target_utterance_for_alignment]

        alignment_tokens_idx = [enc.encode(tu) for tu in target_utterance_for_alignment]

        context_tokens = sum([enc.encode(h) + [EOS_ID] for h in history],[]) 
        context_tokens = [context_tokens for _ in range(args.num_samples)]

        # print(f"Context : {context_tokens}")
        original_sentence = sample(model=model,args=args, context=context_tokens, device=device,
                            repetition_penalty=args.repetition_penalty, weighed_tokens=alignment_tokens_idx)
        # original_sentence, perturb_sentence, _, loss, _ = weight_decoder(model=model, enc=enc, 
        #                                                                             args=args, context=context_tokens,
        #                                                                             device=device,repetition_penalty=args.repetition_penalty,
        #                                                                             classifier=classifier.classifier_head,#knowledge=starter["knowledge"])
        #                                                                 knowledge=None)
        # print(f"Original : {original_sentence.tolist()}")
        spk_turn = {"text":original_sentence.tolist()}
        # print("Check decode: ", enc.decode(cut_seq_to_eos(original_sentence.tolist())))
        hypotesis, _, _ = scorer(args,spk_turn,classifier,enc,class2idx,knowledge=None,plot=False)
        print(f"Hypothesis : {hypotesis}")
        text = hypotesis[0][-1]
        text = " ".join(tokenize.sent_tokenize(text)[:2])
        # print(text_sent)
        # print(text_sent[0])
        print(f"SYS >>> {text}")
        history.append(text)
        history = history[-(2*args.max_history+1):]