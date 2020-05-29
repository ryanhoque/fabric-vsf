"""
Visualization tool for Visual MPC for verifying correctness of vis dynamics and cost fn
"""
import pickle
import cv2
import numpy as np

class Viz():
    def __init__(self, num_trajectories=10, horizon=5, cost_type="L2"):
        self.grid = np.zeros((num_trajectories, horizon+1, 56, 56, 3), dtype=np.uint8) # assumes 56x56
        self.costs = np.zeros(num_trajectories) # index corresponds to trajectory index in grid
        self.num_trajectories = num_trajectories
        self.horizon = horizon
        self.cost_type = cost_type

    def set_context(self, img):
        self.context = img[:,:,:3].astype(np.uint8)

    def set_goal(self, img):
        self.goal = img[:,:,:3].astype(np.uint8)

    def set_grid(self, imgs, acts, costs):
        """
        imgs: num_trajectories x horizon x 56 x 56 x [3-4] np array
        acts: num_trajectories x horizon x 4 np array
        """
        for i in range(self.num_trajectories):
            self.grid[i,0] = self.project_action(self.context, acts[i,0])
            for j in range(1,self.horizon):
                self.grid[i,j] = self.project_action(imgs[i,j-1,:,:,:3].astype(np.uint8), acts[i,j])
            self.grid[i,self.horizon] = imgs[i,self.horizon-1,:,:,:3]
        self.costs = costs


    def project_action(self, img, action):
        # Assumes 56x56 img is not domain randomimzed and clip_act_space=True in cfg
        cx, cy = (action[0]+1)/2, (action[1]+1)/2
        start_point = (int(7 + cx*42), int(48 - cy*42)) # convert to pixel space
        end_point = (int(7 + (cx+action[2])*42), int(48 - (cy+action[3])*42))
        color = (0,0,0)
        thickness = 2
        return cv2.arrowedLine(np.copy(img), start_point, end_point, color, thickness, tipLength=0.1)

    def render_image(self, filepath):
        pixels_per_row = 56 + 2*5 # add buffer
        img = np.ones(((self.num_trajectories+1) * pixels_per_row, max(5,self.horizon+2) * pixels_per_row + 10, 3), dtype=np.uint8) * 255
        # first row: context image, goal image, cost type
        img[5:61, 5:61] = self.context
        img[5:61, self.horizon*pixels_per_row+5:self.horizon*pixels_per_row+61] = self.goal
        cv2.putText(img, "Current", (pixels_per_row+5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        cv2.putText(img, "Goal", ((self.horizon-1)*pixels_per_row+10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        cv2.putText(img, "%s Cost" % self.cost_type, ((self.horizon+1)*pixels_per_row,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        # the rest
        for i in range(1,self.num_trajectories+1):
            for j in range(self.horizon+1):
                img[i*pixels_per_row+5:i*pixels_per_row+61, j*pixels_per_row+5:j*pixels_per_row+61] = self.grid[i-1,j]
            # print cost
            org = ((self.horizon+1)*pixels_per_row, i*pixels_per_row+35)
            # TODO FORMAT STRING
            cv2.putText(img, "%.3f" % self.costs[i-1], org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        cv2.imwrite(filepath, img)

    def test(self):
        """simple test for functionality"""
        img = pickle.load(open('goal_imgs/smooth.pkl', 'rb'))
        self.set_context(img)
        self.set_goal(img)
        act = [-1,-1,1,1]
        self.set_grid(np.tile(img, (10,5,1,1,1)), np.tile(act,(10,5,1)), np.ones(10)*1234.56)
        self.render_image('test')

