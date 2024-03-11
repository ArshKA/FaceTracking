from video_processor import process_video
from clustering_algos import cluster_and_plot, cluster_and_plot_gmm
from grouping import group_bbox, group_probabilities, process_predictions
from visualizations import show_output
from load_model import Predictor
import numpy as np

attributes = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
       'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
       'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Gray_Hair',
       'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
       'Mustache', 'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
       'Receding_Hairline', 'Rosy_Cheeks', 'Smiling', 'Straight_Hair',
       'Wavy_Hair', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necktie',
       'Young', 'Facial_Hair']

model = Predictor('run15/epoch_0_loss_77.499.pth', len(attributes))
frames, embeddings, predictions, face_counts = process_video('media/stock_many.mp4', model)

min_num_faces = round(np.percentile(face_counts, 5))

labels = cluster_and_plot(embeddings, min_num_faces, len(frames)//4)

grouped_bbox = group_bbox(frames, labels)


average_preds = group_probabilities(predictions, labels)



attributes = process_predictions(average_preds, attributes)

video_path = 'media/stock_many.mp4'
output_video_path = 'media/stock_many_out.mp4'

show_output(video_path, grouped_bbox, output_video_path, attributes)

