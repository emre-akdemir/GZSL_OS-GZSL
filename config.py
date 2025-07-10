import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB, AWA2, FLO')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--sentement_embedding', default='sent')
parser.add_argument('--syn_num', type=int, default=300, help='number features to generate per class. 309 for clustering %98')
parser.add_argument('--syn_num_c', type=int, default=304, help='number features to generate per class')
parser.add_argument('--zsl', action='store_true', default=False, help='enable zero-shot learning')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--os_gzsl', action='store_true', default=False, help='enables open set generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features') 
parser.add_argument('--sentSize', type=int, default=1024, help='size of sentement semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector') 
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=125, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ') 
parser.add_argument('--feed_lr', type=float, default=0.00001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.0001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--encoded_noise', action='store_true', default=True, help='enables validation mode')
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
parser.add_argument('--gammaD', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument('--gammaG_D2', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument('--gammaD2', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument("--latent_size", type=int, default=312)
parser.add_argument("--conditional", action='store_true',default=True)
parser.add_argument('--finetune', action='store_true', default=True, help='enables finetuned res101')
parser.add_argument('--sent', action='store_true', default=True, help='enables sentement semantic features')
parser.add_argument('--hybrid', action='store_true', default=True, help='enables semantic attribute and sentement features')
parser.add_argument('--a1', type=float, default=1.0)
parser.add_argument('--a2', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=0.01, help='recons_weight for decoder')
parser.add_argument('--feedback_loop', type=int, default=2)
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')
#openset
parser.add_argument('--nclass_openset', type=int, default=35, help='number of all classes') #35-15     25-25
parser.add_argument('--mixup', action='store_true', default=False, help='enables mixup for open set images')
parser.add_argument('--lam', type=float, default=0.2, help='lambda for mixup')
#clustering
parser.add_argument('--dbscan_', action='store_true', default=False, help='enables DBSCAN')
parser.add_argument('--eps', type=int, default=10, help='eps parameter for DBSCAN')
parser.add_argument('--min_samples', type=int, default=5, help='min sapmles parameter for DBSCAN')
parser.add_argument('--pca', action='store_true', default=False, help='enables PCA')
parser.add_argument('--k_means', action='store_true', default=True, help='enables percentile closest with kmeans')
#parser.add_argument('--k_means_visual', action='store_true', default=True, help='enables percentile closest with kmeans for visual feature')
parser.add_argument('--k', type=int, default=50, help='number of clusters')
parser.add_argument('--percentile_closest', type=int, default=98, help='percentile closest')
parser.add_argument('--syn_num_closest', type=int, default=15200, help='number features to generate per class after percentile closest')
#file
parser.add_argument('--excel_filename', default='D:/CUB/resultsss.xlsx')

opt = parser.parse_args()
opt.lambda2 = opt.lambda1
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize
                
if opt.sent:
    if opt.hybrid:
        opt.attSize = opt.sentSize + opt.attSize 
    else:
        opt.attSize = opt.sentSize

if opt.finetune:
    opt.image_embedding = 'res101_finetuned'
        
opt.latent_size = opt.attSize
opt.nz = opt.attSize


'''
AWA2

parser.add_argument('--ontology_embedding', default='o2v-awa')
parser.add_argument('--syn_num', type=int, default=1800, help='number features to generate per class 1830 for clustering %99'')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=85, help='size of the latent z vector')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.0001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--manualSeed', type=int, default=9182, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument("--latent_size", type=int, default=85)
parser.add_argument('--ontology', action='store_true', default=False, help='enable ontology semantic features')
parser.add_argument('--hybrid', action='store_true', default=True, help='enable hybrid features')
parser.add_argument('--finetuned', action='store_true', default=True, help='enable hybrid features')
parser.add_argument('--nclass_openset', type=int, default=5, help='number of all classes 7-5')
parser.add_argument('--k', type=int, default=10, help='number of clusters')
parser.add_argument('--percentile_closest', type=int, default=99, help='percentile closest')
parser.add_argument('--syn_num_closest', type=int, default=24320, help='number features to generate per class after percentile closest')

if opt.ontology:
    opt.attSize = 100
    if opt.hybrid:
        opt.attSize = 185
'''

'''
FLO

parser.add_argument('--sentement_embedding', default='data')
parser.add_argument('--syn_num', type=int, default=1200, help='number features to generate per class. 1263 for clustering %97)
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--sentSize', type=int, default=1024, help='size of sentement semantic features')
parser.add_argument('--nz', type=int, default=1024, help='size of the latent z vector')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=0.00001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.0001, help='learning rate to train softmax classifier')
parser.add_argument('--manualSeed', type=int, default=806, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=102, help='number of all classes')
parser.add_argument("--latent_size", type=int, default=1024)
parser.add_argument('--nclass_openset', type=int, default=14, help='number of all classes') #14-6     10-10
parser.add_argument('--k', type=int, default=150, help='number of clusters')
parser.add_argument('--percentile_closest', type=int, default=97, help='percentile closest')
parser.add_argument('--syn_num_closest', type=int, default=15200, help='number features to generate per class after percentile closest')
'''



