from video_processor import process_video
from clustering_algos import cluster_and_plot, cluster_and_plot_gmm
from grouping import group_bbox, group_probabilities, process_predictions
from visualizations import show_output
import numpy as np

frames, embeddings, predictions, face_counts = process_video('media/stock_many.mp4')

min_num_faces = round(np.percentile(face_counts, 5))

labels = cluster_and_plot(embeddings, min_num_faces, len(frames)//4)

grouped_bbox = group_bbox(frames, labels)

average_preds = group_probabilities(predictions, labels)


print(embeddings.shape, predictions.shape, face_counts, labels.shape)

print(grouped_bbox)

print(average_preds)

attributes = process_predictions(average_preds)

video_path = 'media/stock_many.mp4'
output_video_path = 'media/stock_many_out.mp4'

show_output(video_path, grouped_bbox, output_video_path, attributes)

