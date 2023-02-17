from nltk.translate.meteor_score import meteor_score
import argparse
 
parser = argparse.ArgumentParser(description='test')
 
parser.add_argument('--hypo', type=str, required=True, help='path to hypo')
parser.add_argument('--ref', type=str, required=True, help='path to ref')

args = parser.parse_args()

with open(args.hypo,'r') as inf:
    hypos = inf.readlines()
with open(args.ref,'r') as inf:
    refs = inf.readlines()
avg_score=0
for hypo,ref in zip(hypos, refs): 

  sent_score = meteor_score([ref],hypo)
  avg_score += sent_score

avg_score = avg_score/(len(refs))
print("{:.4f}".format(avg_score))