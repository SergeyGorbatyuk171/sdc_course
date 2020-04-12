import numpy as np
import scipy as sp
import scipy.spatial

import transforms as ts

class Edge(object):
    '''
    Base edge class.
    Some methods are abstract and should be redefined.
    Provides numerical Jacobian computations.
    '''

    def __init__(self, vertices):
        self.vertices = vertices
        self._J = None
        self.inf = None
    
    def linearize(self):
        DELTA = 1E-9
        self._J = []
        for vertex in self.vertices:
            J = None
            start_params = vertex.params
            for dim in range(vertex.dim):
                vertex.params = start_params
                delta_params = np.zeros(vertex.dim)
                delta_params[dim] = DELTA
                vertex.update(delta_params)
                self.compute_error()
                error_diff = self.error
                vertex.params = start_params
                delta_params[dim] = -DELTA
                vertex.update(delta_params)
                self.compute_error()
                error_diff -= self.error
                if J is None:
                    J = np.zeros((len(error_diff), vertex.dim))
                J[:, dim] = error_diff / 2.0 / DELTA
            self._J.append(J)

        
    def J(self, vertex_index):
        assert self._J is not None and vertex_index < len(self._J)
        return self._J[vertex_index]
    
    @property
    def inf(self):
        raise Exception('Not implemented')
    
    @property
    def error(self):
        raise Exception('Not implemented')
    
    def compute_error(self):
        raise Exception('Not implemented')
    
    def chi2(self):
        return np.dot(np.dot(self.error, self.inf), self.error)


class Vertex(object):
    '''
    Basic vertex class.
    Update method is abstract and should be reimplemented
    '''

    def __init__(self, params):
        self.params = params
    
    def update(self, delta):
        raise Exception('Not implemented')
    
    @property
    def dim(self):
        return len(self.params)


class SE2Vertex(Vertex):
    '''Vertex class that represents SE(2) class of transformations'''

    def __init__(self, params):
        assert len(params) == 3
        super(SE2Vertex, self).__init__(params)
        
    def update(self, delta):
        transform = np.array([np.cos(self.params[2]), -np.sin(self.params[2]), 0.0,
                              np.sin(self.params[2]), np.cos(self.params[2]), 0.0,
                              0.0, 0.0, 1.0]).reshape((3, 3))
        self.params += np.dot(transform, delta)


class Feature(object):
    '''Represents feature and associated data: feature vertex, related edges and type'''
    
    UNDEFINED = 0
    POINT = 1
    LINE = 2
    
    def __init__(self, vertex, edges, ftype):
        self.vertex = vertex
        self.edges = edges
        self.ftype = ftype

    @property
    def visualization_data(self):
        if self.ftype == Feature.POINT:
            return self.vertex.params
        return None


class PriorEdge(Edge):
    inf = None
    error = None
    '''
    #########################################
    TO_IMPLEMENT Seminar.Task#2
    '''
    def __init__(self, vertex, event, cov_diag):
        super(PriorEdge, self).__init__([vertex])
        self.event = event
        self.inf = np.linalg.inv(np.diag(cov_diag))

    def compute_error(self):
        self.error = np.array(self.event['pose']) - np.array(self.vertices[0].params)


class OdometryEdge(Edge):
    inf = None
    error = None

    def __init__(self, from_vertex, to_vertex, event):
        super(OdometryEdge, self).__init__([from_vertex, to_vertex])
        alpha = np.array(event['alpha']).reshape((3, 2))
        self.inf = np.diag(1.0 / np.dot(alpha, np.square(np.array(event['command']))))
        self._v, self._w = event['command']
    
    @property
    def from_vertex(self):
        return self.vertices[0]
    
    @property
    def to_vertex(self):
        return self.vertices[1]
    
    def compute_error(self):
        '''
        #########################################
        TO_IMPLEMENT Seminar.Task#3
        '''

        x_prev, y_prev, orientation_prev = self.from_vertex.params
        x_curr, y_curr, orientation_curr = self.to_vertex.params

        rel_translation = np.array([x_curr - x_prev, y_curr - y_prev])
        projection = np.array([np.cos(orientation_prev), np.sin(orientation_prev)])
        cosine_distance = scipy.spatial.distance.cosine(rel_translation, projection)
        EPS = 1e-3
        if cosine_distance < EPS or np.fabs(cosine_distance - 2.) < EPS:
            # linear
            control_est = np.array([np.dot(rel_translation, projection), 0.])
        else:
            # rotation
            rel_translation *= -1.
            diff_x, diff_y = rel_translation
            mu = 0.5 * (diff_x * np.cos(orientation_prev) + diff_y * np.sin(orientation_prev)) / \
                       (diff_y * np.cos(orientation_prev) - diff_x * np.sin(orientation_prev))
            mid = (self.from_vertex.params + self.to_vertex.params)[:-1] / 2
            center = mid + mu * np.array([y_prev - y_curr, x_curr - x_prev])
            center_in_prev = center - self.from_vertex.params[:-1]
            r = np.linalg.norm(center_in_prev)
            xc, yc = center
            delta_orientation = np.arctan2(y_curr - yc, x_curr - xc) - np.arctan2(y_prev - yc, x_prev - xc)
            control_est = np.array([r * delta_orientation * np.sign(center_in_prev[1]), delta_orientation])

        control_measured = np.array([self._v, self._w])
        self.error = np.zeros(3)
        self.error[:-1] = control_measured - control_est
        self.error[-1] = orientation_curr - orientation_prev - control_est[1]


class Landmark(Vertex):
    '''
    Represents positon of feature in the map
    '''
    
    def __init__(self, params):
        assert len(params) == 2
        super(Landmark, self).__init__(params)
    
    def update(self, delta):
        self.params += delta


class LandmarkObservationEdge(Edge):
    inf = None
    error = None

    '''
    #########################################
    TO_IMPLEMENT Homework.Task#1
    '''
    def __init__(self, pose_vertex, feature_vertex, event):
        super(LandmarkObservationEdge, self).__init__([pose_vertex, feature_vertex])
        Q = np.array(event['Q']).reshape(2, 2)
        self.pose_vertex = pose_vertex
        self.feature_vertex = feature_vertex
        self.inf = np.linalg.inv(Q)
        self.measurement = np.array(event['measurement'])

    def compute_error(self):
        T = ts.Transform2D.from_pose(self.pose_vertex.params)
        self.error = T.transform(self.measurement) - self.feature_vertex.params
