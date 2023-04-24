import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import numpy as np


# Build up a list of quads from input delaunay triangles, returns an Nx4 list of indices on the points used.
def getAllQuads(tri):
    pairings = set()
    quads = []
    # Build up a list of all possible unique quads from triangle neighbor pairings. 
    # In general the worst common case with a fully visible board is 6*6*2=36 triangles, each with 3 neighbor
    # so around ~100 quads.
    for i, neighbors in enumerate(tri.neighbors):
        for k in range(3): # For each potential neighbor (3, one opposing each vertex of triangle)
            nk = neighbors[k]
            if nk != -1:
                # There is a neighbor, create a quad unless it already exists in set
                pair = (i, nk)
                reverse_pair = (nk, i)
                if reverse_pair not in pairings:
                    # New pair, add and create a quad
                    pairings.add(pair)
                    b = tri.simplices[i]
                    d = tri.simplices[nk]                
                    nk_vtx = (set(d) - set(b)).pop()
                    insert_mapping = [2,3,1]
                    b = np.insert(b,insert_mapping[k], nk_vtx)
                    quads.append(b)
    return np.array(quads)

def countHits(given_pts, x_offset, y_offset):
    # Check the given integer points (in unity grid coordinate space) for matches
    # to an ideal chess grid with given initial offsets
    pt_set = set((a,b) for a,b in given_pts)
    [X,Y] = np.meshgrid(np.arange(7) + x_offset,np.arange(7) + y_offset)
    matches = 0
    # count matching points in set
    matches = sum(1 for x,y in zip(X.flatten(), Y.flatten()) if (x,y) in pt_set)
    return matches
        
def getBestBoardMatchup(given_pts):
    best_score = 0
    best_offset = None
    for i in range(7):
        for j in range(7):
            # Offsets from -6 to 0 for both
            score = countHits(given_pts, i-6, j-6)
            if score > best_score:
                best_score = score
                best_offset = [i-6, j-6]
    return best_score, best_offset


def scoreQuad(quad, pts, prevBestScore=0):
    idealQuad = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), idealQuad)
    warped_to_ideal = cv2.perspectiveTransform(np.expand_dims(quad.astype(float),0), M)
    # Warp points and score error
    pts_warped = cv2.perspectiveTransform(np.expand_dims(pts.astype(float),0), M)[0,:,:]
    
    # Get their closest idealized grid point
    pts_warped_int = pts_warped.round().astype(int)
    
    # Count matches
    score, offset = getBestBoardMatchup(pts_warped.round().astype(int))
    if score < prevBestScore:
        return score, None, None, None
    
    # Sum of distances from closest integer value for each point
    # Use this error score for tie-breakers where number of matches is the same.
    error_score = np.sum(np.linalg.norm((pts_warped - pts_warped_int), axis=1))
    
    return score, error_score, M, offset

def brutesacChessboard(xcorner_pts):
    # Build a list of quads to try.
    quads = getAllQuads(tri)
    
    # For each quad, keep track of the best fitting chessboard.
    best_score = 0
    best_error_score = None
    best_M = None
    best_quad = None
    best_offset = None
    for quad in xcorner_pts[quads]:
        score, error_score, M, offset = scoreQuad(quad, xcorner_pts, best_score)
        if score > best_score or (score == best_score and error_score < best_error_score):
            best_score = score
            best_error_score = error_score
            best_M = M
            best_quad = quad
            best_offset = offset

    return best_M, best_quad, best_offset, best_score, best_error_score

quads = getAllQuads(tri)
print(len(quads))

inlier_pts, outlier_pts, pred_pts, final_predictions, prediction_levels, tri, simplices_mask = RunExportedMLOnImage.processImage(gray)
