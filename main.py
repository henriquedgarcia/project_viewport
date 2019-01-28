#!/usr/bin/python3
import configparser
import csv
import sys
import math as m
import numpy as np
import cv2


def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    config = dict(config)

    config['fov'] = dict(x=int(config['input']['fov_x']),
                         y=int(config['input']['fov_y']))

    config['res'] = dict(x=int(config['input']['res_x']),
                         y=int(config['input']['res_y']))

    return config


def main():
    try:
        name = sys.argv[1]
    except IndexError:
        name = 'data/ride_user01_orientation.csv'

    config = parse_config('config.ini')

    csv_reader = csv.reader(open(name, 'r'))
    input_header = next(csv_reader)

    column_names = find_column_name(config, input_header)

    sample_count = 0
    for line in csv_reader:
        yaw = float(line[column_names['yaw']])
        pitch = float(line[column_names['pitch']])
        roll = float(line[column_names['roll']])
        # yaw = 0
        # pitch = 0
        # roll = 0

        if config['input']['unit'] == 'deg':
            yaw = np.deg2rad(yaw)
            pitch = np.deg2rad(pitch)
            roll = np.deg2rad(roll)

        position = dict(yaw=yaw, pitch=pitch, roll=roll)

        # print('Criando objeto')
        viewport_cartesian = Viewport(fov=config['fov'], res=config['res'])

        # print('Rodando')
        viewport_cartesian.rotate(position)

        # print('Projetando para equirretangular')
        equirectangular_map = viewport_cartesian.to_erp()


        im_name = 'output/img/ride_user01_' + str(sample_count) + '.png'
        print(im_name)
        cv2.imwrite(im_name, equirectangular_map)

        sample_count += 1


def find_column_name(config, input_header):
    yaw_column_name = config['input']['yaw_column']
    pitch_column_name = config['input']['pitch_column']
    roll_column_name = config['input']['roll_column']

    yaw_column = input_header.index(yaw_column_name)
    pitch_column = input_header.index(pitch_column_name)
    roll_column = input_header.index(roll_column_name)

    column_names = dict(yaw=yaw_column,
                        pitch=pitch_column,
                        roll=roll_column)

    return column_names


class Viewport:
    def __init__(self, fov, res):
        """
        fov = field-of-vision
        self.x_range = resolução do viewport
        self.y_range =
        self.viewport_cartesian = O viewport em coordenadas cartesianas
        self.viewport_spherical = O viewport em coordenadas polares

        Args:
            fov (dict): field of vision em graus
        """

        self.fov = fov
        self.res = res
        self.viewport_cartesian = self._make_viewport()
        self.viewport_spherical = self.viewport_cartesian.copy()

    @staticmethod
    def show_viewport(viewport):
        # Normalizando
        minimo = np.min(viewport)
        viewport = viewport - minimo
        maximo = np.max(viewport)
        ref = maximo/255
        if not ref == 0:
            viewport = np.round(viewport / ref)
        viewport = viewport.astype(dtype=np.uint8, copy=False)

        print('Min = {}, Max = {}'.format(minimo, maximo + minimo))
        cv2.imshow('image', viewport)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _make_viewport(self):
        """
        Faz o viewport e o coloca tangente a esfera no ponto x=0, y=0, z=1
        (A posição central é o eixo y)
        Return:
            None
        """

        fov = dict(x=np.deg2rad(self.fov['x']),
                   y=np.deg2rad(self.fov['y']))

        res = self.res

        viewport_cartesian = np.ones((3, res['y'], res['x']), dtype=np.double)

        # values = lambda d: np.linspace(-np.tan(fov[d]/2), np.tan(fov[d]/2),
        # res[d])
        x_val = np.linspace(-np.tan(fov['x']/2), np.tan(fov['x']/2), res['x'])
        y_val = np.linspace(-np.tan(fov['y']/2), np.tan(fov['y']/2), res['y'])

        for x in range(res['x']):
            viewport_cartesian[1, :, x] = y_val

        for y in range(res['y']):
            viewport_cartesian[0, y, :] = x_val

        # Viewport.show_viewport(viewport_cartesian[0, :, :])
        # Viewport.show_viewport(viewport_cartesian[1, :, :])
        # Viewport.show_viewport(viewport_cartesian[2, :, :])

        return viewport_cartesian

    @staticmethod
    def _make_rot_matrix(position):
        yaw = position['yaw']  # y fixo, eixo x e z se movem
        pitch = position['pitch']  # x fixo, eixo y e z se movem
        roll = position['roll']  # z fixo, eixo x e y se movem

        m_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                        [0, 1, 0],
                        [-np.sin(yaw), 0, np.cos(yaw)]])

        m_p = np.array([[1, 0, 0],
                        [0, np.cos(pitch), -np.sin(pitch)],
                        [0, np.sin(pitch), np.cos(pitch)]])

        m_r = np.array([[np.cos(roll), -np.sin(roll), 0],
                        [np.sin(roll), np.cos(roll), 0],
                        [0, 0, 1]])

        r_matrix = dict(r_y=m_y,
                        r_p=m_p,
                        r_r=m_r)

        return r_matrix

    @staticmethod
    def _cart2hcs(xyz):
        """
        Horizontal coordinate system
        Args:
            xyz:

        Returns:

        """
        [x, y, z] = xyz
        r = 1
        elevation = -m.atan2(y, m.sqrt(x*x + z*z))  # theta
        azm = m.atan2(x, z)  # phi
        return r, elevation, azm

    def rotate(self, new_position):
        res = self.res
        viewport_cartesian = self.viewport_cartesian
        r_matrix = Viewport._make_rot_matrix(new_position)

        for y in range(res['y']):
            for x in range(res['x']):
                actual_position = viewport_cartesian[:, y, x]
                trans1 = r_matrix['r_r'] @ actual_position
                trans2 = r_matrix['r_p'] @ trans1
                trans3 = r_matrix['r_y'] @ trans2
                viewport_cartesian[:, y, x] = trans3

        self.viewport_cartesian = viewport_cartesian

    def _conv_esf(self):
        viewport_cartesian = self.viewport_cartesian
        viewport_e = self.viewport_spherical
        for y in range(self.res['y']):
            for x in range(self.res['x']):
                hcs = self._cart2hcs(viewport_cartesian[:, y, x])
                [r, elevation, azimuth] = hcs
                viewport_e[0, y, x] = np.rad2deg(azimuth)
                viewport_e[1, y, x] = np.rad2deg(elevation)
                viewport_e[2, y, x] = r
        self.viewport_spherical = viewport_e

    def to_erp(self):

        self._conv_esf()

        viewport_e = self.viewport_spherical
        res = self.res  # pix

        mapa = np.zeros([res['y'], res['x']])

        for y in range(res['y']):
            for x in range(res['x']):
                x_ = (180 + viewport_e[0, y, x]) * res['x'] / 360
                x_ = x_ % res['x']
                y_ = (90 - viewport_e[1, y, x]) * res['y'] / 180
                x_ = np.uint16(np.floor(x_))
                y_ = np.uint16(np.floor(y_))
                y_ = y_ % res['y']
                mapa[y_, x_] = 255
        # Viewport.show_viewport(mapa)
        return mapa


if __name__ == "__main__":
    main()