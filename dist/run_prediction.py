# This file is part of the program for paper 'Simultaneous prediction
# of protein secondary structure population and intrinsic disorder
# using multi-task deep learning'.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os, sys
import psiblast
import cross_validation
import InputData as input_data
import deepMulti, deepSingle
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=22)
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', action="store", default='', dest='input', required=True, help='Input file in FASTA format. ')
parser.add_argument('-o', action="store", default='', dest='output', help='Output file for storing generated results. ')
parser.add_argument('-v', action="store_true", default=False, dest="visualise", help='Visualise the results. '
                                                                  'If not set, no visualisation is generated. ')
parser.add_argument('-f', action="store_true", default='', dest="figure_output", help='The output path of generated visualisation. '
                                                                  'In either .pdf or .png format. Only available when -v is set. ')
parser.add_argument('-s', action="store_true", default=False, dest="single", help='Genearate additional '
                                                                    'results from single task framework (DeepS2D-D) '
                                                                    'for IDP/IDR prediction as a comparison. ')

results = parser.parse_args()

aa_dict = dict()

aa_dict['A']=3.0
aa_dict['C']=2.0
aa_dict['E']=6.0
aa_dict['D']=5.0
aa_dict['G']=4.0
aa_dict['F']=-6.0
aa_dict['I']=-9.0
aa_dict['H']=-3.0
aa_dict['K']=8.0
aa_dict['M']=-8.0
aa_dict['L']=-7.0
aa_dict['N']=-2.0
aa_dict['Q']=7.0
aa_dict['P']=0.0
aa_dict['S']=-1.0
aa_dict['R']=9.0
aa_dict['T']=1.0
aa_dict['W']=-5.0
aa_dict['V']=-10.0
aa_dict['Y']=-4.0

def readFasta(path):
    ids, seqs = [], []
    current_id, current_seq = '', ''
    for line in open(path):
        if line.startswith('>'):
            if current_id != '':
                ids.append(current_id)
                seqs.append(current_seq)
            current_id = line[1:].strip()
        elif line.strip() != '':
            current_seq += line.strip()
    if current_id!='':
        ids.append(current_id)
        seqs.append(current_seq)

    return ids, seqs

def readLines(path):
    ret =[]
    for line in open(path, 'r'):
        ret.append(line.strip())
    return ret

def construct(lines, is_pssp=False):
    lines = [l.strip() for l in lines]
    protein_name = ''
    seq = ''
    sds = []
    pssm = []
    target = ''
    prots, seqs, sdss, pssms, targets = [], [], [], [], []
    for line in lines:
        if line.startswith('#'):
            continue
        elif line.startswith('>'):
            if protein_name != '':
                prots.append(protein_name)
                assert (len(seq) == len(sds))
                seqs.append(seq)
                pssms.append(pssm)
                sdss.append(sds)
                targets.append(target)
            protein_name = line[2:].split(';')[0].split(' ')[0]
            seq = ''
            sds = []
            target = ''
            pssm = []
        else:
            comps = line.strip().split('\t')
            seq += comps[1]
            vals = comps[3].strip().split(' ')
            vals = [float(val) for val in vals]
            vals.insert(0, float(comps[2]))
            vals.insert(0, float(comps[0]))
            pssm.append(vals)
            label = []
            if is_pssp:
                label.append(float(comps[-5]))
                label.append(float(comps[-4]))
                label.append(float(comps[-3]))
            else:
                vv = int(comps[-1])
                if vv == -1:
                    vv = 2
                label.append(vv)
            sds.append(label)
            target += comps[-1]
    prots.append(protein_name)
    assert (len(seq) == len(sds))
    seqs.append(seq)
    pssms.append(pssm)
    sdss.append(sds)
    targets.append(target)
    # writeRaw(output_dir, prots, seqs, pssms, sdss, targets)
    return prots, seqs, targets, np.array(pssms), np.array(sdss)

def construct_features(ids, seqs):
    lines = []
    for index, id in enumerate(ids):
        sys.stdout.write("\r%d completed. " % (index+1))
        sys.stdout.flush()
        lines.append('> '+id+'\n')
        seq = seqs[index]
        pssms = psiblast.getBlast_individual_(id, seq)
        for ind, vec in enumerate(pssms):
            if ind == 0 or vec.strip() == '':
                continue
            comps = vec.strip().split()
            label = '-1'
            line = comps[0]+'\t'+comps[1]+'\t'+str(aa_dict[comps[1]])+'\t'+' '.join(comps[2:])+'\t'+label+'\n'
            lines.append(line)
    print('\n')
    ids, seqs, targets, pssms, sdss = construct(lines)
    dataset = input_data.TestData(ids, seqs, pssms, sdss, window_size=9)
    ids, seqs, feature, label = dataset.getData()
    return ids, seqs, cross_validation.DataSet(feature, label, is_test=True), np.concatenate(label, axis=0)

def run_multi(sess, test_model, test):
    output_diso = []
    output_pssp = []
    all_y = []
    data = test.all()
    y_conv_diso, y_conv_pssp = sess.run([test_model.output_1, test_model.output_2],
                      feed_dict={test_model.input: data[0],
                                 test_model.keep_prob: 1.0,
                                 test_model.keep_prob_2: 1.0})
    # print y_conv_diso
    output_diso.extend(y_conv_diso[:, 1])
    output_pssp.extend(y_conv_pssp)
    all_y.extend(data[1])
    return output_diso, output_pssp, all_y

def run_single(sess, test_model, test):
    output = []
    all_y = []
    data = test.all()
    # print len(data), len(data[0]), len(data[1])
    y_conv = sess.run(test_model.output,
                      feed_dict={test_model.input: data[0],
                                 test_model.keep_prob: 1.0})

    output.extend(y_conv[:, 1])
    all_y.extend(data[1])
    return output, all_y

def visualise(ids, seqs, disos, pssp_helxes, pssp_strands, pssp_coils,
              compared_disos=None,
              names=None):
    fig = plt.figure(figsize=(18, 6*len(ids)))
    for i, id in enumerate(ids):
        ax1 = fig.add_subplot(len(ids), 1, i+1)
        index = np.arange(len(seqs[i]))
        # print index

        p1= ax1.bar(index, pssp_helxes[i], 0.5,
                         alpha=0.8,
                         color='b',
                         label='Helix (H)',
                         edgecolor="b")

        p2 = ax1.bar(index, pssp_strands[i], 0.5,
                         alpha=0.8,
                         color='g',
                         label='Strands (E)',
                         edgecolor="g")

        p3, = ax1.plot(np.array(disos[i]),
                         color='r',
                         label='Disordered (M)', linewidth=2)

        p4 = None
        if compared_disos != None:
            p4, = ax1.plot(np.array(compared_disos[i]),
                             color='black',
                             label='Disordered (S)', linewidth=2)

        p5 = ax1.bar(index, np.array(pssp_coils[i]), 0.5, alpha=0.5, color='grey', label='Coil (C)',
                         edgecolor="grey")

        ax1.plot([0.5]*len(seqs[i]), color='grey', linestyle='--', linewidth=2)

        ax1.set_xlabel('Residues', size=26)
        ax1.set_ylabel('Predicted PSSP/IDR', size=26)
        name = id
        if names != None:
            name = names[i]
        plt.title('Predicted results for '+name, size=26)

        ax1.set_ylim([0, 1.2])
        ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0.0', '0.25', '0.5', '0.75', '1.0'])

        ax1.set_xlim([-10, len(seqs[i])+10])
        aas = [str(aa) for aa in seqs[i]]
        # ax1.set_xticks(index)
        # ax1.set_xticklabels(aas)
        ax1.set_xticks(np.arange(min(index), max(index)+1, 20))

        # ax1.setgrid()
        if compared_disos != None:
            ax1.legend([p1, p2, p3, p4, p5], ['Helix', 'Strands', 'IDR (Multi-task)', 'IDR (Single-task)', 'Coil'],
                       fancybox=True, framealpha=0.5, ncol=5, bbox_to_anchor=(0.958, 1), fontsize=20)
        else:
            ax1.legend([p1, p2, p3, p5], ['Helix', 'Strands', 'IDR (Multi)', 'Coil'],
                       fancybox=True, framealpha=0.5, ncol=5, fontsize=20)
    plt.tight_layout()
    # plt.show()
    if results.figure_output != '':
        plt.savefig(results.figure_output, dpi=350)
        print('PROGRESS: Figure saved to '+results.figure_output)
    else:
        plt.savefig('../figures/visualisation.pdf', dpi=350)
        print('PROGRESS: Figure saved to ../figures/visualisation.pdf')


def predict():
    input = results.input
    output = results.output
    if output != '':
        output_file = open(output, 'w')
    else:
        output_file = sys.stdout
    is_plot = results.visualise
    is_compared = results.single

    ids, seqs = readFasta(input)
    # ids = ['P04637', 'Q9BQ15']
    # names = ['p53', 'SOSSB1']
    # seqs = ['MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD',
    #         'MTTETFVKDIKPGLKNLNLIFIVLETGRVTKTKDGHEVRTCKVADKTGSINISVWDDVGNLIQPGDIIRLTKGYASVFKGCLTLYTGRGGDLQKIGEFCMVYSEVPNFSEPNPEYSTQQAPNKAVQNDSNPSASQPTTGPSAASPASENQNGNGLSAPPGPGGGPHPPHTPSHPPSTRITRSQPNHTPAGPPGPSSNPVSNGKETRRSSKR',
    #         ]
    print("PROGRESS: Generating PSSMs... ")
    ids, seqs, test, label = construct_features(ids, seqs)

    compared_disos = None
    # if is_compared=True, also predict IDP/IDR with singletask model, which will be compared to the results generated from multask models
    if is_compared:
        test_model_compared = deepSingle.deepSingle(is_train=False, is_context=False)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, '../models/checkpoints/best_diso.ckpt')
            print("PROGRESS: Model restored: DeepS2D-D. ")
            output_diso, labels = run_single(sess, test_model_compared, test)
            count = 0
            compared_disos = []
            output_file.write("Results of DeepS2D-D for IDP/IDR prediction. "+'\n')
            output_file.write("ID\tPos\tAA\tScore\tPrediction"+'\n')
            for i, id in enumerate(ids):
                diso = []
                for j, aa in enumerate(seqs[i]):
                    predict_diso = 0
                    if output_diso[count] >= 0.5:
                        predict_diso = 1
                    diso.append(output_diso[count])
                    output_file.write(id+'\t'+str(j+1) + '\t'+ aa+'\t'+ str('{0:.3f}'.format(output_diso[count]))
                                      +'\t'+str(predict_diso)+'\n')
                    count += 1
                compared_disos.append(diso)
        print("PROGRESS: Prediciton completed for DeepS2D-D. ")
        output_file.write('\n')

    new_graph = tf.Graph()
    with  tf.Session(graph=new_graph) as sess:
        test_model = deepMulti.deepMulti(is_train=False, is_context=False)
        saver = tf.train.Saver()
        saver.restore(sess, '../models/checkpoints/best_multi.ckpt')
        print("PROGRESS: Model restored: Multitask-D.")
        output_diso, output_pssp, labels = run_multi(sess, test_model, test)
        count = 0
        disos, pssp_helxes, pssp_strands, pssp_coils = [], [], [], []
        output_file.write("Results of Multitask-D for IDP/IDR prediction. "+'\n')
        output_file.write("ID\tPos\tAA\tIDP/IDR Score\tIPR/IDR predict\tP.Helix\tP.Strands\tP.Coils"+'\n')
        for i, id in enumerate(ids):
            diso, helix, strands, coils = [], [], [], []
            for j, aa in enumerate(seqs[i]):
                predict_diso = 0
                if output_diso[count] >= 0.5:
                    predict_diso = 1
                diso.append(output_diso[count])
                helix.append(output_pssp[count][0])
                strands.append(output_pssp[count][1])
                coils.append(output_pssp[count][2])
                output_file.write(id+'\t'+str(j+1) + '\t'+aa+'\t'+str('{0:.3f}'.format(output_diso[count]))+'\t'+str(predict_diso)+'\t'
                      +str('{0:.3f}'.format(output_pssp[count][0]))+'\t'
                      +str('{0:.3f}'.format(output_pssp[count][1]))+'\t'
                      +str('{0:.3f}'.format(output_pssp[count][2]))+'\n')
                count += 1
            disos.append(diso)
            pssp_helxes.append(helix)
            pssp_strands.append(strands)
            pssp_coils.append(coils)
        print("PROGRESS: Prediciton completed for Multitask-D. ")

        if is_plot:
            visualise(ids, seqs, disos, pssp_helxes, pssp_strands, pssp_coils, compared_disos=compared_disos)


if __name__ == '__main__':

    predict()




