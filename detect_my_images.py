import argparse
import datetime
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib as mpl
mpl.use('tkagg')

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
# This is needed since the .py is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



#accounting to the time to make the inside folder
dt = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
def get_output_dir(out_dir):
    #create directory
    defaultDir = 'detect_img_out'
    if out_dir == None or out_dir == '': #for default 
        if not os.path.exists(defaultDir):
            os.mkdir(defaultDir) #create base folder: /detect_img_out
        out_dir = defaultDir+'/'+dt #inside folder: /detect_img_out/2019-08-16_11_11_11
        os.mkdir(out_dir)
    else: #for specific directory
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir = out_dir
        
    
    return out_dir
    


orgimg_list = []    
#load the image path to array
#store the img list
def load_images_to_array(inDirectory):    
    for img in os.listdir(inDirectory):

        if os.path.splitext(img)[-1] == '.jpg' or os.path.splitext(img)[-1] == '.jpeg' or os.path.splitext(img)[-1] == '.JPG' or os.path.splitext(img)[-1] == '.png' or os.path.splitext(img)[-1] == '.PNG':
            orgimg_list.append(img)

#Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

#tensorflow 
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

  def tohex(array):
    array = np.asarray(array, dtype='uint32')
    return ((array[:, :, 0]<<16) + (array[:, :, 1]<<8) + array[:, :, 2])

if __name__ == '__main__':
    
    if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
        raise ImportError('Please upgrade your TensorFlow installation to > v1.12.*.')
    
    parser = argparse.ArgumentParser(description="load images")
    parser.add_argument('-i', '--inDir', type=str, required=True, help='Directory containing the images')
    parser.add_argument('-o','--outDir',type=str,required=False, help='Out put directory for the detected images')
    #parser.add_argument('-s', '--imgSize', type=int, nargs=2, required=False, metavar=('width', 'height'), help='Image size. default:2736*1824')
    args = parser.parse_args()
    load_images_to_array(args.inDir)
    input_directory = args.inDir #get the images path with folder
    output_Directory = get_output_dir(args.outDir) #make output images dir
    print('*************************Total ',len(orgimg_list),' images in here*************************')
    print('Loading..........')
    
    ## To change font size of label:
    #In file models/research/object_detection/utils/visualization_utils.py starting from line 208:
    #use full path
    #font = ImageFont.truetype('/Users/shingwaichan/venv/lib/python3.6/site-packages/tensorflow/models/research/object_detection/utils/Arial.ttf', 24)
    #except IOError:
    #font = ImageFont.load_default()

    
    # This is needed to display the images in jupyter notebook.
    #%matplotlib inline

    #my own model for land and machine
    #inference_graph_model_2019_08_12_Land_Machine 800x600
    MODEL_NAME = 'inference_graph_2019_08_16_2736x1824'
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = 'training/labelmap.pbtxt'
    detection_graph = tf.Graph()
    
    ## Load a (frozen) Tensorflow model into memory.
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    ## Loading label map
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    ##Detection
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = input_directory
    TEST_IMAGE_PATHS = []
    for orgimg in orgimg_list:
        TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, orgimg))

    print('image_list = ',TEST_IMAGE_PATHS)

    image_count = 1
    print('******************detection start******************')
    #image list for return
    return_dic = {}
    for image_path in TEST_IMAGE_PATHS:
        print('processing image: ',image_count,'/ ',len(TEST_IMAGE_PATHS),'.....','\nimage path:',image_path)
        image = Image.open(image_path)
        IMAGE_DPI = 72.0
        dpi = 'dpi' in image.info.keys()
        if dpi:
            if image.info['dpi']:
                IMAGE_DPI = float(image.info['dpi'][0])
        print('iamge size =',image.width,'x',image.height,'dpi =',IMAGE_DPI)
        # Size, in inches, of the output images.
        #DPI Calculator / PPI Calculator: https://mitblog.pixnet.net/blog/post/37708222
        IMAGE_SIZE = (image.width/IMAGE_DPI, image.height/IMAGE_DPI)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        if image.format == "PNG":
            #sRGB convert to RGB
            image = image.convert('RGB')
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)#To change the line width of boxes: thickness=4 (change to number what you want) deafult is 4 

        plt.figure(figsize=IMAGE_SIZE)
        plt.rcParams['figure.dpi'] = IMAGE_DPI
        #plt.imshow(image_np)
        plt.axis('off')
        #plt.savefig('./image_out/img_out{}.jpg'.format(image_count))
        #In jupyter notebook
        #this function must have the GPU to show the image, while cannot show it. 
        #But we can use this #mpl.use('tkagg') to make it.  
        #plt.show()
        #get the box coordinates
        boxes = output_dict['detection_boxes']
        # get all boxes from an array
        max_boxes_to_draw = boxes.shape[0]
        # get scores to get a threshold
        scores = output_dict['detection_scores']
        #Accuracy rate default 0.5
        min_score_thresh=.5
        #image array to store the box frame eg:"image_name1":[{"Land": [0.36901385, 0.2333157, 0.5195253, 0.3745013]}...]
        image_list = [] 
        # iterate over all objects found
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                class_name = category_index[output_dict['detection_classes'][i]]['name']
                
                #print(class_name,':\t','[',round(boxes[i][0]*img_height,5),'\t',round(boxes[i][1]*img_width,5),'\t',round(boxes[i][2]*img_height,5),'\t',round(boxes[i][3]*img_width,5),']', output_dict['detection_classes'][i])
                
                #output_dict['detection_boxes']: ymin, xmin, ymax, xmax
                # y Start, x Start, y End , x End 
                img_dic = {class_name: [boxes[i][0]*image.height,boxes[i][1]*image.width,boxes[i][2]*image.height,boxes[i][3]*image.width]}
                image_list.append(img_dic)
                print(img_dic)
            return_dic[orgimg_list[image_count-1]] = image_list #add array to Dictionary
            
        #change format to image 
        im = Image.fromarray(image_np) 
        #im.save(os.path.join(output_Directory, os.path.basename(orgimg_list[image_count-1])))
        out_path = output_Directory+'/'+orgimg_list[image_count-1]
        im.save(out_path)
        print('image',image_count,'/',len(orgimg_list),'finished.....')
        print('Output directory: ',out_path)

        image_count+=1
        #im.show() #show in the photo browser
    print('******************Detection complete.******************')
    
    #dictionary to json
    import json
    #numpy.float32 can not dirct cover to json need this encoder
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)
    #cover to json
    json_out_dir = output_Directory + '/result.json'
    with open(json_out_dir,'w') as fp:
        j = json.dumps(return_dic,cls=MyEncoder)
        fp.write(j) #write to file
    print('Json result stored in', json_out_dir,'\n',j)
    sys.exit(1)


