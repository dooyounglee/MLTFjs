#학습
#꽃사진 http://download.tensorflow.org/example_images/flower_photos.tgz
#https://github.com/tensorflow/hub/tree/master/examples/image_retraining
#pip3 install pillow

#python retrain.py
#	--bottleneck_dir=./workspace/bottlenecks #학습할 사진을 인셉션용 학습 데이터로 변환해서 저장해둘 디렉터리
#	--model_dir=./workspace/inception #인셉션 모델을 내려받을 경로
#	--output_graph=./workspace/flowers_graph.pb #학습된 모델(.pb)을 저장할 경로
#	--output_labels=./workspace/flowers_labels.txt #레이블 이름들을 저장해둘 파일 경로
#	--image_dir ./workspace/flower_photos #원본 이미지 경로
#	--how_many_training_steps 1000 #반복 학습 횟수

#예측 predict.py
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

tf.app.flags.DEFINE_string("output_graph","./workspace/flowers_graph.pb","학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string("output_labels","./workspace/flowers_labels.pb","학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_boolean("show_image",True-,"이미지 추론 후 이미지를 보여줍니다.")

FLAGS = tf.app.flags.FLAGS

def main(_):
	labels = [line.rstrip() for line in tf.gfile.Gfile(FLAGS.output_labels)]
	
	with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
		graph_def = tf.GraphDef()
		graph_def.ParseFormString(fp.read())
		tf.import_graph_def(graph_def,name='')
	
	with tf.Session() as sess:
		logits = sess.graph.get_tensor_by_name('final_result:0')
		
		image = tf.gfile.FastGFile(sys.argv[1],'rb').read()
		prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})

	print('=== 예측 결과 ===')
	for i in range(len(labels)):
		name = labels[i]
		score = prediction[0][i]
		print('%s (%.2f%%)' % (name, score*100))
	
	if FLAGS.show_image:
		img = mpimg.imread(sys.argv[1])
		plt.imshow(img)
		plt.show()

if __name__=="__main__":
	tf.app.run()

#C:\>python predict.py workspace/flower_photos/roses/3065719996_c16ecd5551.jpg