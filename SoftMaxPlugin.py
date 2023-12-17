import PyPluMA
import PyIO
import torch.nn.functional as F
from sklearn import metrics
import pickle
import pandas as pd

def assign_score(score_dict, all_capri_complexes):
    score = []
    for ppi in all_capri_complexes:
        model_id = ppi.split('_')[0]
        score.append(score_dict[model_id])
    return score

def readDict(inputfile):
    d = dict()
    infile = open(inputfile, 'r')
    for line in infile:
        if (len(line) > 1):
           line = line.strip()
           contents = line.split(',')
           d[contents[0]] = contents[1]
    return d

def read_score_dict(data_dir, score_file_affix, unique_pids, print_labels=False):
    """
    Read the scores file performed by DeepRank
    """
    score_dict = {}
    statistics_dict={"Sample":[], "N_pos":[], "N_neg":[]}
    for pid in unique_pids:
        score_file = f"{data_dir}/{pid}/{pid}.{score_file_affix}"
        all_scores = []
        for line in open(score_file).readlines():
            curr_model, score = line.strip('\n').split('\t')
            model_id = curr_model.split('_')[0] + '-' + curr_model.split('_')[1]
            score = float(score)
            score_dict[model_id] = score
            all_scores.append(score)
        if print_labels:
            print(f"Sample {pid}: {int(np.sum(all_scores))} positives out of {len(all_scores)} models")
            statistics_dict['Sample'].append(pid)
            statistics_dict['N_pos'].append(int(np.sum(all_scores)))
            statistics_dict['N_neg'].append(len(all_scores) - int(np.sum(all_scores)))
    if print_labels:
        df=pd.DataFrame(statistics_dict)
        df.to_csv("capri_dataset.csv")

    return score_dict


class SoftMaxPlugin:
    def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
    def run(self):
        pass
    def output(self, outputfile):
     output2file = open(PyPluMA.prefix()+"/"+self.parameters["dataset"], "rb")
     output2 = pickle.load(output2file)
     # pred_probabilities = F.softmax(output, dim=1)
     pred_probabilities = list(output2.cpu().detach().numpy())

     # Select probability of the complex to be positive (y=1)
     #pred_probabilities = [float(x[1]) for x in output]

     labels_dict = readDict(PyPluMA.prefix()+"/"+self.parameters["labelsfile"])
     # assign integer label
     for key_i in labels_dict.keys():
       labels_dict[key_i] = int(float(labels_dict[key_i]))
     testlistfile = open(PyPluMA.prefix()+"/"+self.parameters["testlist"], "rb")
     test_list_updated = pickle.load(testlistfile)
     unprocessedfile = open(PyPluMA.prefix()+"/"+self.parameters["unprocessed"], "rb")
     unprocessed_complexes = pickle.load(unprocessedfile)

     # Create labels array
     labels = []
     for ppi in test_list_updated:
       model_id = ppi.split('_')[0]
       labels.append(labels_dict[model_id])

     # append unprocessed complexes
     pred_probabilities = pred_probabilities + [1 for x in list(unprocessed_complexes)]
     all_capri_complexes = test_list_updated + list(unprocessed_complexes)
     true_labels = labels + [labels_dict[x.split('_')[0]] for x in list(unprocessed_complexes)]

     df = dict()
     df[self.parameters["root"]] = pred_probabilities
     
     tools = PyIO.readParameters(PyPluMA.prefix()+"/"+self.parameters["toolsfile"])
     for key in tools:
         df[key] = assign_score(readDict(PyPluMA.prefix()+"/"+tools[key]), all_capri_complexes)
     df["label"] = true_labels
     
     #dove_scores = assign_score(dove_dict, all_capri_complexes)
     #deeprank_scores = assign_score(deeprank_dict, all_capri_complexes)
     #haddockScore_scores = assign_score(haddockScore_dict, all_capri_complexes)
     #iScore_scores = assign_score(iScore_dict, all_capri_complexes)
     uniquepidfile = open(PyPluMA.prefix()+"/"+self.parameters["uniquepid"], "rb")
     unique_pids = pickle.load(uniquepidfile)
     capri_labels = assign_score(read_score_dict(PyPluMA.prefix()+"/"+self.parameters["qualitydir"], self.parameters["qualityname"], unique_pids), all_capri_complexes) 
     #capri_dict = read_score_dict(CAPRI_DIR, 'capri', unique_pids)
     #capri_labels = assign_score(capri_dict, all_capri_complexes)
     df["capri_quality"] = capri_labels
     df["model_name"] = all_capri_complexes
     df = pd.DataFrame(df)
     df['target'] = df['model_name'].apply(lambda x: x.split('-')[0])
     df['pid'] = df['model_name'].apply(lambda x: x.split('_')[0])
     outfile = open(outputfile, "wb")
     pickle.dump(df, outfile)
