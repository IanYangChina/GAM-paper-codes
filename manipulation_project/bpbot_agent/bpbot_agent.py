import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from bpbot.binpicking import *
from bpbot.tangle_solution import LineDetection, EntanglementMap
from bpbot.config import BinConfig
from bpbot.utils import normalize_depth_map


class BPBot(object):
    def __init__(self, image_size=480, depth_max_distance=300, depth_min_distance=180, depth_rescale=1000):
        self.path = os.path.dirname(__file__)
        self.cfg = BinConfig(os.path.join(self.path, "cfg/config.yaml"))
        self.image_size = image_size
        self.depth_max_distance = depth_max_distance
        self.depth_min_distance = depth_min_distance
        self.depth_rescale = depth_rescale

    def detect_and_draw(self, real_depth):
        img = self.processed_depth(real_depth)
        img_edge, num_edge = self.detect_edge(img)
        emap = self.get_entanglement_map(img)
        grasps, img_input, img_grasp = self.detect_grasp_point(img, n_grasp=10)
        t_grasps, t_img_input, t_img_grasp = self.detect_nontangle_grasp_point(img, n_grasp=10)

        # # visulization
        fig = plt.figure()

        fig.add_subplot(231)
        plt.imshow(img_input, cmap='gray')
        plt.title("Depth")

        fig.add_subplot(232)
        plt.imshow(img_edge)
        plt.axis("off")
        plt.title("Edge")

        fig.add_subplot(233)
        plt.imshow(img_input)
        plt.imshow(cv2.resize(emap, (img_input.shape[1], img_input.shape[0])), interpolation='bilinear', alpha=0.4,
                   cmap='jet')
        plt.title("Depth + EMap")

        fig.add_subplot(234)
        plt.imshow(img_grasp)
        plt.axis("off")
        plt.title("FGE")

        fig.add_subplot(235)
        plt.imshow(t_img_grasp)
        plt.axis("off")
        plt.title("FGE + EMap")

        # grasp related
        fig.add_subplot(236)
        plt.imshow(t_img_grasp)
        plt.imshow(cv2.resize(emap, (img_input.shape[1], img_input.shape[0])), interpolation='bilinear', alpha=0.3,
                   cmap='jet')
        plt.axis("off")
        plt.title("FGE + EMap (EMap shown)")

        plt.tight_layout()
        # plt.get_current_fig_manager().full_screen_toggle()
        plt.show()

    def processed_depth(self, real_depth):
        """
        :param real_depth: real depth map obtained by mujoco --> from manipulation_project.env.utils import get_real_depth_map
        :return: 3-channel depth image required by the bpbot package
        """
        depth = normalize_depth_map(real_depth * self.depth_rescale,
                                    self.depth_max_distance, self.depth_min_distance,
                                    self.image_size, self.image_size)
        return np.concatenate([[depth], [depth], [depth]]).transpose((1, 2, 0))

    def detect_edge(self, img):
        (length_thre, distance_thre, sliding_size, sliding_stride, c_size) = self.cfg.t_params
        norm_img = cv2.resize(adjust_grayscale(img), (c_size, c_size))

        ld = LineDetection(length_thre=length_thre, distance_thre=distance_thre)
        lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img, vis=True)
        return drawn, lines_num

    def get_entanglement_map(self, img):
        (length_thre, distance_thre, sliding_size, sliding_stride, c_size) = self.cfg.t_params
        norm_img = cv2.resize(adjust_grayscale(img), (c_size, c_size))

        em = EntanglementMap(length_thre, distance_thre, sliding_size, sliding_stride)
        emap, wmat_vis, w, d = em.entanglement_map(norm_img)
        lmap = em.line_map(norm_img)
        bmap = em.brightness_map(norm_img)
        return emap

    def detect_grasp_point(self, img, n_grasp=10):
        """Detect grasp point using graspability
        Parameters:
            n_grasp {int} -- number of grasps you want to output
            img_path {str} -- image path
            g_params {tuple} -- graspaiblity parameters
            h_params {tuple} -- hand (gripper) parameters
            margins {tuple} -- crop roi if you need
        Returns:
            grasps {list} -- grasp candidates [g,x,y,z,a,rot_step, depth_step]
            img {array} -- cropped input image
            drawn {array} -- image that draws detected grasps
        """
        img_adj = adjust_grayscale(img)
        cropped_height, cropped_width, _ = img.shape
        (finger_w, finger_h, open_w, gripper_size) = self.cfg.h_params
        gripper = Gripper(finger_w=finger_w,
                          finger_h=finger_h,
                          open_w=open_w,
                          gripper_size=gripper_size)

        hand_open_mask, hand_close_mask = gripper.create_hand_model()

        (rstep, dstep, hand_depth) = self.cfg.g_params
        method = Graspability(rotation_step=rstep,
                              depth_step=dstep,
                              hand_depth=hand_depth)

        # generate graspability map
        main_proc_print("Generate graspability map  ... ")
        candidates = method.graspability_map(img_adj,
                                             hand_open_mask=hand_open_mask,
                                             hand_close_mask=hand_close_mask)

        if candidates != []:
            # detect grasps
            main_proc_print(f"Detect grasp poses from {len(candidates)} candidates ... ")
            grasps = method.grasp_detection(
                candidates, n=n_grasp, h=cropped_height, w=cropped_width)
            if grasps != []:
                important_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
                # draw grasps
                drawn = gripper.draw_grasp(grasps, img_adj.copy(), top_idx=0)
                # cv2.imshow("window", drawn)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                return grasps, img_adj, drawn
            else:
                warning_print("Grasp detection failed! No grasps!")
                return None, img, img

        else:
            warning_print("Grasp detection failed! No grasps!")
            return None, img, img

    def detect_nontangle_grasp_point(self, img, n_grasp=10):
        """Detect grasp point using graspability
        Parameters:
            n_grasp {int} -- number of grasps you want to output
            img_path {str} -- image path
            g_params {tuple} -- graspaiblity parameters
            h_params {tuple} -- hand (gripper) parameters
            t_params {tuple} -- entanglemet map parameters
            margins {tuple} -- crop roi if you need
        Returns:
            grasps {list} -- grasp candidates [g,x,y,z,a,rot_step, depth_step]
            img {array} -- cropped input image
            drawn {array} -- image that draws detected grasps
        """
        emap = self.get_entanglement_map(img)

        (finger_w, finger_h, open_w, gripper_size) = self.cfg.h_params
        gripper = Gripper(finger_w=finger_w,
                          finger_h=finger_h,
                          open_w=open_w,
                          gripper_size=gripper_size)

        (rstep, dstep, hand_depth) = self.cfg.g_params
        method = Graspability(rotation_step=rstep,
                              depth_step=dstep,
                              hand_depth=hand_depth)

        cropped_height, cropped_width, _ = img.shape
        hand_open_mask, hand_close_mask = gripper.create_hand_model()

        main_proc_print("Generate graspability map  ... ")
        candidates = method.combined_graspability_map(img, hand_open_mask, hand_close_mask, emap)

        if candidates != []:
            # detect grasps
            main_proc_print(f"Detect grasp poses from {len(candidates)} candidates ... ")
            grasps = method.grasp_detection(
                candidates, n=n_grasp, h=cropped_height, w=cropped_width)

            if grasps != [] :
                important_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
                # draw grasps
                drawn = gripper.draw_grasp(grasps, img.copy(), top_idx=0)
                # cv2.imshow("window", drawn)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                return grasps, img, drawn
            else:
                warning_print("Grasp detection failed! No grasps!")
                return None, img, img
        else:
            warning_print("Grasp detection failed! No grasps!")
            return None, img, img


if __name__ == "__main__":
    import timeit

    start = timeit.default_timer()

    main()

    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))
