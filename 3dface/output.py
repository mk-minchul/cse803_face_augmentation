import numpy as np
import os, sys
from glob import glob 
import scipy.io as sio 
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.transform import rescale, resize
import argparse
import time
import ast
import math
import subprocess

sys.path.append('../face3d/')
import face3d
from face3d import mesh

from api import PRN
import random

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture

def transform_test(vertices, triangles, colors, obj, camera, h = 112, w = 112):
	'''
	Args:
		obj: dict contains obj transform paras
		camera: dict contains camera paras
	'''
	R = mesh.transform.angle2matrix(obj['angles'])
	transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])
	
	if camera['proj_type'] == 'orthographic':
		projected_vertices = transformed_vertices
		image_vertices = mesh.transform.to_image(projected_vertices, h, w)
	else:

		## world space to camera space. (Look at camera.) 
		camera_vertices = mesh.transform.lookat_camera(transformed_vertices, camera['eye'], camera['at'], camera['up'])
		## camera space to image space. (Projection) if orth project, omit
		projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near = camera['near'], far = camera['far'])
		## to image coords(position in image)
		image_vertices = mesh.transform.to_image(projected_vertices, h, w, True)

	rendering = mesh.render.render_colors(image_vertices, triangles, colors, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 1)
	return rendering


def main(args):
    if args.isShow or args.isTexture:
        import cv2
        from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)
    errfile = open('errors.txt', 'w')
    # ------------- load data
    image_dir = args.inputDir
    save_dir = args.outputDir
    ssubdirs = []
    if args.infile is None:
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        ssubdirs = os.listdir(image_dir)
    else:
        file = open(args.infile, "r")
        ssubdirs = file.read().split("\n")
   
    
    for j, dir_path in enumerate(ssubdirs):
        print("Current directory: ", dir_path)
        image_folder = os.path.join(image_dir, dir_path)
        save_folder = os.path.join(save_dir, dir_path)
        if not os.path.exists(save_folder): os.mkdir(save_folder)

        types = ('*.jpg')
        image_path_list= []
        image_path_list.extend(glob(os.path.join(image_folder, types)))
        total_num = len(image_path_list)
        for i, image_path in enumerate(image_path_list):
            #stime = time.perf_counter()
            name = image_path.strip().split('/')[-1][:-4]
            # read image
            image = imread(image_path)
            [h, w, c] = image.shape
            if c>3:
                image = image[:,:,:3]

            # the core: regress position map
            if args.isDlib:
                max_size = max(image.shape[0], image.shape[1])
                if max_size> 1000:
                    image = rescale(image, 1000./max_size)
                    image = (image*255).astype(np.uint8)
                pos = prn.process(image) # use dlib to detect face
                if pos is None:
                    print(name, 'no face', file = errfile)
                    continue

            else:
                if image.shape[0] == image.shape[1]:
                    image = resize(image, (256,256))
                    pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
                else:
                    box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
                    pos = prn.process(image, box)
            
            image = image/255.
            if pos is None:
                continue

            if args.is3d or args.isMat or args.isPose or args.isShow:
                # 3D vertices
                vertices = prn.get_vertices(pos)
                if args.isFront:
                    save_vertices = frontalize(vertices)
                else:
                    save_vertices = vertices.copy()
                save_vertices[:,1] = h - 1 - save_vertices[:,1]


            if args.is3d:
                # corresponding colors
                colors = prn.get_colors(image, vertices)

                if args.isTexture:
                    if args.texture_size != 256:
                        pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range = True)
                    else:
                        pos_interpolated = pos.copy()
                    texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
                    if args.isMask:
                        vertices_vis = get_visibility(vertices, prn.triangles, h, w)
                        uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
                        uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range = True)
                        texture = texture*uv_mask[:,:,np.newaxis]
                    #write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
                else:
                    pass
                    #write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

            #if args.isMat:
            #    sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})

            if args.isKpt or args.isShow:
                # get landmarks
                kpt = prn.get_landmarks(pos)
                np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)

            if args.isPose or args.isShow:
                # estimate pose
                camera_matrix, pose = estimate_pose(vertices)
                #np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose) 
                #np.savetxt(os.path.join(save_folder, name + '_camera_matrix.txt'), camera_matrix) 

                #np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)
                if pose[0] > 0.52 or pose[0] < -0.52:
                    print(name, " horz", file = errfile)
                    continue;
                if pose[1] > 0.40 or pose[1] < -0.40:
                    print(name, " vert", file = errfile) 
                    continue;
            
            ncolors = colors/np.max(colors)
            nvertices = save_vertices - np.mean(save_vertices, 0)[np.newaxis, :]
            obj = {}
            camera = {}
            scale_init = 230/(np.max(nvertices[:,1]) - np.min(nvertices[:,1])) # scale face model to real size
            obj['s'] = scale_init
            obj['angles'] = [0, 0, 0]
            obj['t'] = [0, 0, 0]
            # obj: center at [0,0,0]. size:200
            camera['proj_type'] = 'perspective'
            camera['at'] = [0, 0, 0]
            camera['near'] = 1000
            camera['far'] = -100
            # eye position
            camera['fovy'] = 30
            camera['up'] = [0, 1, 0] 
            angle = random.randint(50, 140)
            sign = random.choice([-1,1])
            val = sign*angle
            camera['eye'] = [val, 0, 250] # stay in front
            world_up = np.array([0, 1, 0]) # default direction
            camera['up'] = world_up
            image = transform_test(nvertices,prn.triangles, ncolors, obj, camera) 
            imsave(os.path.join(save_folder, name + 'fr.jpg'), img_as_ubyte(image))
            #etime = time.perf_counter()
            #print(etime - stime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='TestImages/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default=True, type=ast.literal_eval,
                        help='whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default=False, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default=False, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--isPose', default=False, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isShow', default=False, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=False, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isDepth', default=False, type=ast.literal_eval,
                        help='whether to output depth image')
    # update in 2017/4/27
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    parser.add_argument('--isMask', default=False, type=ast.literal_eval,
                        help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
    # update in 2017/7/19
    parser.add_argument('--texture_size', default=256, type=int,
                        help='size of texture map, default is 256. need isTexture is True')

    parser.add_argument('--infile', default=None, type=str,
                        help='File what to process')

    main(parser.parse_args())
