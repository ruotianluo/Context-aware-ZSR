import sys
import pickle
fn = sys.argv[1]
result = pickle.load(open(fn, 'rb'))

# print(result['ausuc']['AUC_val'], result['ausuc']['HM'])
print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t' %(
    result['ungen_target']['mcls_ac'], result['ungen_target']['mins_ac'],\
    result['gen_target']['mcls_ac'], result['gen_target']['mins_ac'],\
    result['ungen_source']['mcls_ac'], result['ungen_source']['mins_ac'],\
    result['gen_source']['mcls_ac'], result['gen_source']['mins_ac']))
# print(result['ungen_target']['precs'])
# print(result['gen_target']['precs'])