#!/usr/bin/python3
# python/example3.py -- running NanoGUI in detached mode
# (contributed by Dmitriy Morozov)
#
# NanoGUI was developed by Wenzel Jakob <wenzel@inf.ethz.ch>.
# The widget drawing code is based on the NanoVG demo application
# by Mikko Mononen.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE.txt file.

import argparse
import copy
import gc
import logging
import math
import os
import re
import sys
import time
from enum import IntEnum
from typing import List

import nanogui
import numpy as np
import pyopenvdb as vdb
import pyscreenshot
from fabnn.label_grid import LabelGrid
from nanogui import *

np.set_printoptions(precision=3, suppress=True, linewidth=300, threshold=1500)


def create_logger(app_name):
    """Create a logging interface"""
    logging_level = os.getenv("logging", logging.INFO)
    logging.basicConfig(
        level=logging_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(app_name)
    return logger


def get_timer():
    if sys.platform == "win32":
        # On Windows, the best timer is time.clock
        return time.clock
    else:
        # On most other platforms the best timer is time.time
        return time.time


def lerp(x: float, y: float, interp: float):
    assert 0.0 <= interp <= 1.0
    return x * interp + (1.0 - interp) * y


logger = create_logger("VDB_view")


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ColorCoding(IntEnum):
    NONE = 0
    VALUE = 1
    BOUNDINGBOX = 2

    SHADING = 64


class Camera(object):
    """ represents a camera """

    def __init__(
        self,
        position=[0, 0, 0],
        direction=[0, 0, -1],
        fovY=45.0,
        near=0.1,
        far=1000.0,
        aspectRatio=1.0,
        sensitivity=0.1,
        cutout_origin=(None, None),
        cutout_size=(None, None),
    ):
        self.position = position
        self.direction = direction
        self.fovY = fovY
        self.near = near
        self.far = far
        self.aspectRatio = aspectRatio
        self.up = (0, 1, 0)
        self.sensitivity = sensitivity
        self.cutout = dotdict(
            {
                "origin": cutout_origin,
                "size": cutout_size,
                "active": (cutout_origin[0] is not None and cutout_origin[1] is not None)
                and (cutout_size[0] > 0.0 and cutout_size[1] > 0.0),
            }
        )

        if self.cutout.active:
            self.cutout.origin = 2.0 * np.array(self.cutout.origin) - 1.0
            self.cutout.size = np.array(self.cutout.size) * 2.0

        self.active = False

    def matrix(self):
        return np.dot(self.projection(), self.view())

    def projection(self):
        l = -1
        r = 1
        b = -1
        t = 1
        if self.cutout.active:
            l = self.cutout.origin[0]
            t = 2.0 + self.cutout.origin[1]
            r = l + self.cutout.size[0]
            b = t - self.cutout.size[1]

        halfAngle = self.near * math.tan(math.radians(self.fovY / 2))
        # TODO: self.aspectRatio *
        l *= halfAngle
        r *= halfAngle
        b *= halfAngle
        t *= halfAngle

        return nanogui.frustum(l, r, b, t, self.near, self.far)

    def view(self):
        target = [self.position[i] + self.direction[i] for i in range(len(self.position))]
        return nanogui.lookAt(self.position, target, self.up)

    def right(self):
        return np.cross(self.direction, self.up).tolist()

    def offsetPosition(self, offset):
        self.position = [self.position[i] + offset[i] for i in range(3)]

    def button(self, position, pressed):
        self.active = pressed

    def __str__(self):
        return """Camera:
                Position: {}
                Direction: {}""".format(
            self.position, self.direction
        )

    def motion(self, rel):
        if self.active:
            angles = np.multiply(np.radians(rel), self.sensitivity)
            cos, sin = np.cos(angles), np.sin(angles)

            rotate_y = np.matrix([[cos[0], 0, sin[0]], [0, 1, 0], [-sin[0], 0, cos[0]]])

            rotate_x = np.matrix([[1, 0, 0], [0, cos[1], -sin[1]], [0, sin[1], cos[1]]])

            rotation = np.dot(rotate_y, rotate_x)
            self.direction = rotation.dot(self.direction).tolist()[0]
            return True
        else:
            return False

    def zoomToBox(self, lowerCorner, upperCorner):

        # bounding sphere
        center = np.add(upperCorner, lowerCorner) / 2
        radius = np.linalg.norm(np.subtract(upperCorner, center))

        distance = (radius / math.tan(math.radians(self.fovY / 2.0))) - np.linalg.norm(
            np.subtract(center, self.position)
        )

        self.offsetPosition([-self.direction[i] * distance for i in range(3)])
        self.direction = np.subtract(center, self.position)
        self.direction = np.divide(self.direction, np.linalg.norm(self.direction))


class Light(object):
    def __init__(
        self,
        position=(0.0, 0.0, 0.0, 1.0),
        color=(1.0, 1.0, 1.0),
        attenuation=1.0,
        ambientCoefficient=0.1,
        coneAngle=0.0,
        coneDirection=(0.0, 0.0, 0.0),
    ):
        self.position = position
        self.color = color
        self.attenuation = attenuation
        self.ambientCoefficient = ambientCoefficient
        self.coneAngle = coneAngle
        self.coneDirection = coneDirection

    def __repr__(self):
        return " ".join(
            [
                str(self.__getattribute__(key))
                for key in [
                    "position",
                    "color",
                    "attenuation",
                    "ambientCoefficient",
                    "coneAngle",
                    "coneDirection",
                ]
            ]
        )


class PointLight(Light):
    def __init__(self, position, color=(1.0, 1.0, 1.0), attenuation=1.0, ambientCoefficient=0.1):
        position = [position[0], position[1], position[2], 1.0]
        super(PointLight, self).__init__(
            position=position,
            color=color,
            attenuation=attenuation,
            ambientCoefficient=ambientCoefficient,
        )


class DirectionalLight(Light):
    def __init__(self, direction, color=(1.0, 1.0, 1.0), ambientCoefficient=0.1):
        direction = [direction[0], direction[1], direction[2], 0.0]
        super(DirectionalLight, self).__init__(
            position=direction,
            color=color,
            attenuation=0.0,
            ambientCoefficient=ambientCoefficient,
        )


class Cutplane(object):
    def __init__(self, position, normal):
        self.position = position
        self.normal = normal


class VoxelShader(GLShader):
    MAX_LIGHTS = 10
    MAX_COLORS = 10
    MAX_CUTPLANES = 6

    def __init__(self, name):
        super(VoxelShader, self).__init__()

        self.define("MAX_LIGHTS", str(VoxelShader.MAX_LIGHTS))
        self.define("MAX_COLORS", str(VoxelShader.MAX_COLORS))
        self.define("MAX_CUTPLANES", str(VoxelShader.MAX_CUTPLANES))

        for key, value in ColorCoding.__members__.items():
            self.define("COLORCODING_" + key, str(int(value)))

        self.lights = []
        self.cutplanes = []

        self.init(
            # An identifying name
            name,
            # Vertex shader
            """#version 330
            
            in vec3 position;
            in vec3 color;

            out vec3 vertColor;

            void main() {
                vertColor = color;
                gl_Position = vec4(position, 1.0);
            }""",
            # Fragment shader
            """#version 330
            
            uniform mat4 u_model;
            uniform vec3 u_cameraPosition;
            uniform float u_materialShininess;

            uniform int u_numLights;
            uniform struct Light {
                vec4 position;
                vec3 color;
                float attenuation;
                float ambientCoefficient;
                float coneAngle;
                vec3 coneDirection;
            } u_lights[MAX_LIGHTS];

            uniform int u_colorCode;
            uniform vec3 u_valueRangeMin;
            uniform vec3 u_valueRangeMax;
            uniform int u_numColors;
            uniform vec3 u_colorScheme[MAX_COLORS];
            uniform int u_gamma;

            vec3 color(vec3 color) {
                if((u_colorCode & COLORCODING_VALUE) > 0){
                    float value = (color.x - u_valueRangeMin.x) / (u_valueRangeMax.x - u_valueRangeMin.x);
                    int index = max(0, min(int(value * float(u_numColors)), MAX_COLORS - 1));
                    return u_colorScheme[index];
                } else if((u_colorCode & COLORCODING_BOUNDINGBOX) > 0){
                    vec3 value = vec3(0.0);
                    value.x = (color.x - u_valueRangeMin.x) / (u_valueRangeMax.x - u_valueRangeMin.x);
                    value.y = (color.y - u_valueRangeMin.y) / (u_valueRangeMax.y - u_valueRangeMin.y);
                    value.z = (color.z - u_valueRangeMin.z) / (u_valueRangeMax.z - u_valueRangeMin.z);
                    return value;
                } else {
                    return color;
                }
            }

            vec3 ApplyLight(Light light, vec3 surfaceColor, vec3 normal, vec3 surfacePos, vec3 surfaceToCamera) {
                vec3 surfaceToLight;
                float attenuation = 1.0;
                if(light.position.w == 0.0) {
                    //directional light
                    surfaceToLight = normalize(light.position.xyz);
                    attenuation = 1.0; //no attenuation for directional lights
                } else {
                    //point light
                    surfaceToLight = normalize(light.position.xyz - surfacePos);
                    float distanceToLight = length(light.position.xyz - surfacePos);
                    attenuation = 1.0 / (1.0 + light.attenuation * pow(distanceToLight, 2));

                    //cone restrictions (affects attenuation)
                    float lightToSurfaceAngle = degrees(acos(dot(-surfaceToLight, normalize(light.coneDirection))));
                    if(lightToSurfaceAngle > light.coneAngle){
                        attenuation = 0.0;
                    }
                }

                //ambient
                vec3 ambient = light.ambientCoefficient * surfaceColor.rgb * light.color;

                //diffuse
                float diffuseCoefficient = max(0.0, dot(normal, surfaceToLight));
                vec3 diffuse = diffuseCoefficient * surfaceColor.rgb * light.color;
                
                //specular
                float specularCoefficient = 0.0;
                if(diffuseCoefficient > 0.0)
                    specularCoefficient = pow(max(0.0, dot(surfaceToCamera, reflect(-surfaceToLight, normal))), u_materialShininess);
                vec3 specular = specularCoefficient * light.color;

                //linear color (color before gamma correction)
                return ambient + attenuation*(diffuse + specular);
            }

            in vec3 fragVert;
            in vec3 fragNormal;
            in vec3 fragColor;

            out vec4 finalColor;

            void main() {
                vec3 normal = normalize(transpose(inverse(mat3(u_model))) * fragNormal);
                vec3 surfacePos = vec3(u_model * vec4(fragVert, 1));
                vec3 surfaceColor = color(fragColor);
                vec3 surfaceToCamera = normalize(u_cameraPosition - surfacePos);

                vec3 linearColor = vec3(0);
                if((u_colorCode & COLORCODING_SHADING) > 0) {
                    //combine color from all the lights
                    for(int i = 0; i < u_numLights; ++i){
                        linearColor += ApplyLight(u_lights[i], surfaceColor.rgb, normal, surfacePos, surfaceToCamera);
                    }
                } else {
                    linearColor = surfaceColor;
                }

                if(u_gamma>0){
                    linearColor = pow(linearColor, vec3(1./2.2));
                }
                finalColor = vec4(vec3(linearColor), 1.0);
            }""",
            # Geometry Shader
            """#version 150 core

            layout(points) in;
            layout(triangle_strip, max_vertices = 12) out;   

            in vec3 vertColor[];

            uniform mat4 u_model;
            uniform mat4 u_projection;
            uniform vec3 u_cameraPosition;

            uniform int u_numCutplanes;
            uniform struct Cutplane {
                vec3 position;
                vec3 normal;
            } u_cutplanes[MAX_CUTPLANES];

            out vec3 fragVert;
            out vec3 fragNormal;
            out vec3 fragColor;

            void main()
            {
                vec3 worldPosition = (u_model * gl_in[0].gl_Position).xyz;
                vec4 vPos = u_projection * vec4(worldPosition, 1.0);
                float w = vPos.w + length(u_projection * u_model * vec4(1.0, 1.0, 1.0, 1.0));
                if( abs(vPos.z) > w ||
                    abs(vPos.x) > w ||
                    abs(vPos.y) > w )
                    return;

                for(int i = 0; i < u_numCutplanes; i++){
                    if(dot(u_cutplanes[i].normal, worldPosition - u_cutplanes[i].position) < 0)
                        return;
                }
                
                const vec3 corners[] = vec3[](
                        vec3(0.0, 0.0, 0.0),
                        vec3(1.0, 0.0, 0.0),
                        vec3(1.0, 1.0, 0.0),
                        vec3(0.0, 1.0, 0.0),
                        vec3(0.0, 0.0, 1.0),
                        vec3(1.0, 0.0, 1.0),
                        vec3(1.0, 1.0, 1.0),
                        vec3(0.0, 1.0, 1.0));

                vec3 toCamera = normalize(u_cameraPosition - worldPosition);

                int xIndices[4];
                int yIndices[4];
                int zIndices[4];
                vec3 normals[3];

                if(dot(corners[4], toCamera) > 0) {
                    // Z Upface
                    zIndices[0] = 4;
                    zIndices[1] = 5;
                    zIndices[2] = 7;
                    zIndices[3] = 6;
                    normals[2] = corners[4];
                } else {
                    // Z Downface
                    zIndices[0] = 0;
                    zIndices[1] = 1;
                    zIndices[2] = 3;
                    zIndices[3] = 2;
                    normals[2] = -corners[4];
                }

                if(dot(corners[3], toCamera) > 0) {
                    // Y Backface
                    yIndices[0] = 3;
                    yIndices[1] = 2;
                    yIndices[2] = 7;
                    yIndices[3] = 6;
                    normals[1] = corners[3];
                } else {
                    // Y Frontface
                    yIndices[0] = 0;
                    yIndices[1] = 1;
                    yIndices[2] = 4;
                    yIndices[3] = 5;
                    normals[1] = -corners[3];
                }

                if(dot(corners[1], toCamera) > 0) {
                    // X Rightface
                    xIndices[0] = 1;
                    xIndices[1] = 2;
                    xIndices[2] = 5;
                    xIndices[3] = 6;
                    normals[0] = corners[1];
                } else {
                    // X Leftface
                    xIndices[0] = 0;
                    xIndices[1] = 3;
                    xIndices[2] = 4;
                    xIndices[3] = 7;
                    normals[0] = -corners[1];
                }

                for(int i = 0; i < 4; i++){
                    vec3 offset = corners[xIndices[i]];
                    gl_Position = u_projection * u_model * vec4(gl_in[0].gl_Position.xyz + offset, 1.0);
                    fragVert = gl_in[0].gl_Position.xyz + offset;
                    fragNormal = normals[0];
                    fragColor = vertColor[0];
                    EmitVertex();
                }
                EndPrimitive();

                for(int i = 0; i < 4; i++){
                    vec3 offset = corners[yIndices[i]];
                    gl_Position = u_projection * u_model * vec4(gl_in[0].gl_Position.xyz + offset, 1.0);
                    fragVert = gl_in[0].gl_Position.xyz + offset;
                    fragNormal = normals[1];
                    fragColor = vertColor[0];
                    EmitVertex();
                }
                EndPrimitive();

                for(int i = 0; i < 4; i++){
                    vec3 offset = corners[zIndices[i]];
                    gl_Position = u_projection * u_model * vec4(gl_in[0].gl_Position.xyz + offset, 1.0);
                    fragVert = gl_in[0].gl_Position.xyz + offset;
                    fragNormal = normals[2];
                    fragColor = vertColor[0];
                    EmitVertex();
                }
                EndPrimitive();
                
            }
            """,
        )

    def addLight(self, light: Light):
        self.lights.append(light)
        self.__uploadLights()
        return self.lights.index(light)

    def removeLight(self, index: int):
        del self.lights[index]
        self.__uploadLights()

    def addCutplane(self, plane: Cutplane):
        self.cutplanes.append(plane)
        self.__uploadCutplanes()
        return self.cutplanes.index(plane)

    def removeCutplane(self, index: int):
        del self.cutplanes[index]
        self.__uploadCutplanes()

    def editCutplane(self, index: int, plane: Cutplane):
        self.cutplanes[index] = plane
        self.__uploadCutplanes()

    def setColorScheme(self, scheme: list):
        assert len(scheme) <= VoxelShader.MAX_COLORS
        self.setUniform("u_numColors", len(scheme))
        for idx, color in enumerate(scheme):
            self.setUniform("u_colorScheme[{}]".format(idx), color)

    def setGamma(self, value):
        self.setUniform("u_gamma", int(value))

    def update(self):
        self.__uploadLights()
        self.__uploadCutplanes()

    def __uploadLights(self):
        self.setUniform("u_numLights", len(self.lights))
        for idx, light in enumerate(self.lights):
            self.__uploadLight(idx, light)

    def __uploadLight(self, index: int, light: Light):
        if index >= VoxelShader.MAX_LIGHTS:
            raise IndexError(
                "Light index {} out of range [0,{}]".format(index, VoxelShader.MAX_LIGHTS)
            )
        for attribute in light.__dict__.keys():
            logger.debug(
                "{} {}".format(
                    "u_lights[{}].".format(index) + attribute, getattr(light, attribute)
                )
            )
            self.setUniform("u_lights[{}].".format(index) + attribute, getattr(light, attribute))

    def __uploadCutplanes(self):
        self.setUniform("u_numCutplanes", len(self.cutplanes))
        for idx, light in enumerate(self.cutplanes):
            self.__uploadCutplane(idx, light)

    def __uploadCutplane(self, index: int, cutplane: Cutplane):
        if index >= VoxelShader.MAX_CUTPLANES:
            raise IndexError(
                "Cutplane index {} out of range [0,{}]".format(index, VoxelShader.MAX_CUTPLANES)
            )
        for attribute in cutplane.__dict__.keys():
            logger.debug(
                "{} {}".format(
                    "u_cutplanes[{}].".format(index) + attribute, getattr(cutplane, attribute)
                )
            )
            self.setUniform(
                "u_cutplanes[{}].".format(index) + attribute, getattr(cutplane, attribute)
            )


class CorrespondenceShader(VoxelShader):
    def __init__(self, name):
        super(VoxelShader, self).__init__()

        self.define("MAX_LIGHTS", str(VoxelShader.MAX_LIGHTS))
        self.define("MAX_COLORS", str(VoxelShader.MAX_COLORS))
        self.define("MAX_CUTPLANES", str(VoxelShader.MAX_CUTPLANES))

        for key, value in ColorCoding.__members__.items():
            self.define("COLORCODING_" + key, str(int(value)))

        self.lights = []
        self.cutplanes = []

        self.init(
            # An identifying name
            name,
            # Vertex shader
            """#version 330
            
            uniform mat4 u_model;
            uniform mat4 u_projection;
            
            in vec3 position;
            in vec3 color;

            out vec3 vertColor;

            void main() {
                vertColor = color;
                gl_Position = vec4(position, 1.0);
            }""",
            # Fragment shader
            """#version 330

            uniform int u_colorCode;
            uniform vec3 u_valueRangeMin;
            uniform vec3 u_valueRangeMax;
            uniform int u_numColors;
            uniform vec3 u_colorScheme[MAX_COLORS];

            vec3 color(vec3 color) {
                if((u_colorCode & COLORCODING_VALUE) > 0){
                    float value = (color.x - u_valueRangeMin.x) / (u_valueRangeMax.x - u_valueRangeMin.x);
                    int index = max(0, min(int(value * float(u_numColors)), MAX_COLORS - 1));
                    return u_colorScheme[index];
                } else if((u_colorCode & COLORCODING_BOUNDINGBOX) > 0){
                    vec3 value = vec3(0.0);
                    value.x = (color.x - u_valueRangeMin.x) / (u_valueRangeMax.x - u_valueRangeMin.x);
                    value.y = (color.y - u_valueRangeMin.y) / (u_valueRangeMax.y - u_valueRangeMin.y);
                    value.z = (color.z - u_valueRangeMin.z) / (u_valueRangeMax.z - u_valueRangeMin.z);
                    return value;
                } else {
                    return color;
                }
            }

            in vec3 fragColor;

            out vec4 finalColor;

            void main() {
                finalColor = vec4(color(fragColor), 1.0);
            }""",
            # Geometry Shader
            """#version 150 core

            layout(lines) in;
            layout(line_strip, max_vertices = 2) out;   

            in vec3 vertColor[];

            uniform mat4 u_model;
            uniform mat4 u_projection;
            uniform vec3 u_cameraPosition;

            uniform int u_numCutplanes;
            uniform struct Cutplane {
                vec3 position;
                vec3 normal;
            } u_cutplanes[MAX_CUTPLANES];

            out vec3 fragColor;

            void main()
            {
                vec3 worldPosition = (u_model * gl_in[0].gl_Position).xyz;
                vec4 vPos = u_projection * vec4(worldPosition, 1.0);
                float w = vPos.w + length(u_projection * u_model * vec4(1.0, 1.0, 1.0, 1.0));
                if( abs(vPos.z) > w ||
                    abs(vPos.x) > w ||
                    abs(vPos.y) > w )
                    return;

                for(int i = 0; i < u_numCutplanes; i++){
                    if(dot(u_cutplanes[i].normal, worldPosition - u_cutplanes[i].position) < 0)
                        return;
                }

                gl_Position = u_projection * u_model * gl_in[0].gl_Position;
                fragColor = vertColor[0];
                EmitVertex();
                gl_Position = u_projection * u_model * gl_in[1].gl_Position;
                fragColor = vertColor[1];
                EmitVertex();
                EndPrimitive();                
            }
            """,
        )


class TestApp(Screen):
    def __init__(self, args, background=None):
        super(TestApp, self).__init__((1024, 1024), "NanoGUI Test", nSamples=0, fullscreen=False)

        self.filename = args.filename

        self.colorCoding = dotdict(
            {
                "state": ColorCoding.SHADING,
                "valueButton": None,
                "bbButton": None,
                "shadingButton": None,
                "min": (0.0, 0.0, 0.0),
                "max": (0.0, 0.0, 0.0),
                "minTextbox": None,
                "maxTextbox": None,
            }
        )

        window = Window(self, "Menu")
        window.setPosition((15, 15))
        window.setLayout(BoxLayout(Orientation.Vertical, Alignment.Fill, 15, 6))

        Label(window, "Frame time", "sans-bold")
        self.frametime = dotdict({"label": Label(window, "0.0ms"), "lastupdate": get_timer()()})

        b = Button(window, "Print Camera")

        def printCameraCB():
            logger.info(self.camera)
            logger.info(self.camera.matrix())

        b.setCallback(printCameraCB)

        b = Button(window, "Focus Camera")

        def focusCameraCB():
            self.camera.zoomToBox(self.boundingBox[0], self.boundingBox[1])

        b.setCallback(focusCameraCB)

        Label(window, "Color Coding", "sans-bold")
        tools = Widget(window)
        tools.setLayout(BoxLayout(Orientation.Horizontal, Alignment.Middle, 0, 6))
        self.colorCoding.valueButton = ToolButton(tools, entypo.ICON_COLOURS)
        self.colorCoding.valueButton.setTooltip("Tab")

        def colorCodeValueCB(state):
            state = ColorCoding.VALUE if state else ColorCoding.NONE
            self.colorCoding.state = state | (self.colorCoding.state & ColorCoding.SHADING)
            self.__updateColorCoding()

        self.colorCoding.valueButton.setChangeCallback(colorCodeValueCB)

        self.colorCoding.bbButton = ToolButton(tools, entypo.ICON_NOTIFICATION)
        self.colorCoding.bbButton.setTooltip("Tab")

        def colorCodeBBCB(state):
            state = ColorCoding.BOUNDINGBOX if state else ColorCoding.NONE
            self.colorCoding.state = state | (self.colorCoding.state & ColorCoding.SHADING)
            self.__updateColorCoding()

        self.colorCoding.bbButton.setChangeCallback(colorCodeBBCB)

        self.colorCoding.shadingButton = ToolButton(tools, entypo.ICON_LIGHT_BULB)
        self.colorCoding.shadingButton.setTooltip("Light Shading")
        self.colorCoding.shadingButton.setFlags(Button.Flags.ToggleButton)
        self.colorCoding.shadingButton.setPushed(
            bool(self.colorCoding.state & ColorCoding.SHADING)
        )

        def colorCodeShadingCB(state):
            state = ColorCoding.SHADING if state else ColorCoding.NONE
            self.colorCoding.state = state | (
                self.colorCoding.state & (ColorCoding.VALUE + ColorCoding.BOUNDINGBOX)
            )
            self.__updateColorCoding()

        self.colorCoding.shadingButton.setChangeCallback(colorCodeShadingCB)

        b = ToolButton(tools, 0, "γ")
        b.setTooltip("Gamma Correction")
        b.setFlags(Button.Flags.ToggleButton)
        b.setChangeCallback(lambda state: self.shader.bind() or self.shader.setGamma(state))
        b.setPushed(True)

        self.colorCoding.minTextbox = FloatBox(window)
        self.colorCoding.minTextbox.setValueIncrement(0.1)
        self.colorCoding.minTextbox.setFormat("[-]?[0-9]*\\.?[0-9]+")
        self.colorCoding.minTextbox.setEditable(True)
        self.colorCoding.minTextbox.setUnits("Min")
        self.colorCoding.minTextbox.setSpinnable(True)

        def minTextBoxCB(value):
            self.__setMinMax([value] * 3, self.colorCoding.max)

        self.colorCoding.minTextbox.setCallback(minTextBoxCB)

        self.colorCoding.maxTextbox = FloatBox(window)
        self.colorCoding.maxTextbox.setValueIncrement(0.1)
        self.colorCoding.maxTextbox.setFormat("[-]?[0-9]*\\.?[0-9]+")
        self.colorCoding.maxTextbox.setEditable(True)
        self.colorCoding.maxTextbox.setUnits("Max")
        self.colorCoding.maxTextbox.setSpinnable(True)

        def maxTextBoxCB(value):
            self.__setMinMax(self.colorCoding.min, [value] * 3)

        self.colorCoding.maxTextbox.setCallback(maxTextBoxCB)

        b = Button(window, "Reload File", entypo.ICON_CCW)

        def reloadVDB():
            self.vdb.grids = None
            if self.vdb.window:
                self.removeChild(self.vdb.window)
            self.performLayout()
            self.shader.bind()
            gridindex = self.shader.attribVersion("position")
            self.shader.freeAttrib("position")
            self.shader.freeAttrib("color")
            self.num_triangles = 0
            self.showGrid(gridindex)

        b.setCallback(reloadVDB)

        b = Button(window, "Quit")
        b.setTooltip("Esc")

        def quitCB():
            self.setVisible(False)

        b.setCallback(quitCB)

        self.cameraPositions = []
        self.cameraDirections = []
        for pos in getattr(args, "position", ["[-1, 1, 1]"]):
            position = [float(i) for i in re.findall("-?\d+\.?\d*", pos)]
            assert len(position) == 3, "wrong numbers given for a 3D position: {}".format(
                position
            )
            self.cameraPositions.append(position)
        for dir in getattr(args, "direction", ["[1, -1, -1]"]):
            direction = [float(i) for i in re.findall("-?\d+\.?\d*", dir)]
            assert len(direction) == 3, "wrong numbers given for a 3D direction: {}".format(
                direction
            )
            self.cameraDirections.append(direction)

        assert len(self.cameraPositions) == len(
            self.cameraDirections
        ), "Unequal number of positions and directions:\nPositions: {}\nDirections: {}".format(
            self.cameraPositions, self.cameraDirections
        )

        crop = (
            (np.array(args.crop).reshape(2, 2) / self.size()).reshape(-1)
            if hasattr(args, "crop")
            else [None, None, 0, 0]
        )

        if hasattr(args, "crop") and (crop[2] / crop[3]) != self.width() / self.height():
            crop[2] = crop[3] * (self.height() / self.width())
            new_shape = (crop.reshape(2, 2) * self.size()).reshape(-1)
            logger.warn("Aspect ratio doesn't match, fixing crop to: {}".format(new_shape))

        self.camera = Camera(
            self.cameraPositions[0],
            self.cameraDirections[0],
            cutout_origin=crop[:2],
            cutout_size=crop[-2:],
        )

        def resizeCB(xy):
            logger.debug("Resizing Window to {}".format(xy))
            self.camera.aspectRatio = xy[1] / xy[0]

        self.setResizeCallback(resizeCB)

        self.shader = VoxelShader("Voxel shader")
        self.shader.bind()
        self.shader.setGamma(True)

        self.shader.setUniform("u_materialShininess", 200.0)

        # TODO make the light scene dependent
        self.shader.addLight(DirectionalLight([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], 0.1))
        self.shader.addLight(DirectionalLight([1.0, 1.0, -1.0], [0.5, 0.5, 0.5], 0.1))
        self.shader.addLight(DirectionalLight([-1.0, 1.0, 1.0], [0.5, 0.5, 0.5], 0.1))
        self.shader.addLight(DirectionalLight([-1.0, 1.0, -1.0], [0.5, 0.5, 0.5], 0.1))

        colorScheme = [
            (247.0 / 255.0, 252.0 / 255.0, 253.0 / 255.0),
            (229.0 / 255.0, 245.0 / 255.0, 249.0 / 255.0),
            (204.0 / 255.0, 236.0 / 255.0, 230.0 / 255.0),
            (153.0 / 255.0, 216.0 / 255.0, 201.0 / 255.0),
            (102.0 / 255.0, 194.0 / 255.0, 164.0 / 255.0),
            (65.0 / 255.0, 174.0 / 255.0, 118.0 / 255.0),
            (35.0 / 255.0, 139.0 / 255.0, 69.0 / 255.0),
            (0.0 / 255.0, 109.0 / 255.0, 44.0 / 255.0),
            (0.0 / 255.0, 68.0 / 255.0, 27.0 / 255.0),
        ]

        self.shader.setColorScheme(colorScheme)

        self.num_triangles = 0
        self.model_matrix = np.eye(4)

        self.__updateColorCoding()

        self.speed = 0.5

        self.vdb = dotdict({"grids": None, "metadata": None, "window": None, "panel": None})

        self.screenshot_countdown = 0
        self.screenshot_path = ""
        if hasattr(args, "screenshot"):
            if background is not None:
                self.setBackground(nanogui.Color(background, background, background, 255))
            self.screenshot_countdown = 3
            self.screenshot_camera = 0
            base, ext = os.path.splitext(args.screenshot)
            self.screenshot_path = base + "_{}" + ext
            self.screenshot_quit = args.quit

        self.boundingBox = [[0, 0, 0], [0, 0, 0]]
        self.boundingBoxIndex = [[0, 0, 0], [0, 0, 0]]
        self.boundingBoxIndexClipped = [[0, 0, 0], [0, 0, 0]]
        self.keepClipping = False

        self.showGrid(
            args.gridname, not (hasattr(args, "position") and hasattr(args, "direction"))
        )

        self.constructCutplaneWindow()

        self.performLayout()

    #######################
    #   NanoGUI Section   #
    #######################

    def drawContents(self):
        if self.screenshot_countdown:
            self.camera.zoomToBox(self.boundingBox[0], self.boundingBox[1])
            state = ColorCoding.NONE
            self.colorCoding.state = state | (
                self.colorCoding.state & (ColorCoding.VALUE + ColorCoding.BOUNDINGBOX)
            )
            self.__updateColorCoding()
            for widget in self:
                widget.setVisible(False)

        if self.shader is not None:
            self.shader.bind()

            gl.Enable(gl.DEPTH_TEST)

            model = self.model_matrix
            projection = self.camera.matrix()

            # print(model)
            # print(self.camera.view())
            # print(self.camera.projection())
            # print(projection)

            # print(nanogui.project((0,0,1), model, projection, (512,512)))

            # mvp[0:3, 0:3] *= 0.25
            # mvp[0, :] *= self.size()[1] / self.size()[0]

            self.shader.setUniform("u_model", model)
            self.shader.setUniform("u_projection", projection)
            self.shader.setUniform("u_cameraPosition", self.camera.position, False)
            self.shader.drawArray(
                gl.LINES if type(self.shader) is CorrespondenceShader else gl.POINTS,
                0,
                self.num_triangles,
            )

            now = get_timer()()
            frametime = now - self.frametime.lastupdate
            self.frametime.label.setCaption(
                "{: >4.1f}ms {: >3.0f}FPS".format(frametime * 1000, 1.0 / frametime)
            )
            self.frametime.lastupdate = now

        super(TestApp, self).drawContents()
        if self.screenshot_countdown:
            self.screenshot_countdown -= 1
            if self.screenshot_countdown == 0:
                region = [0, 0, *self.size()]
                logger.info("Screenshot of {}".format(region))
                screenshot = pyscreenshot.grab(region)
                screenshot.save(self.screenshot_path.format(self.screenshot_camera))
                self.screenshot_camera += 1
                if self.screenshot_camera < len(self.cameraPositions):
                    self.camera.position = self.cameraPositions[self.screenshot_camera]
                    self.camera.direction = self.cameraDirections[self.screenshot_camera]
                    self.screenshot_countdown = 3
                else:
                    # took screenshots of all cameras, clean up
                    if self.screenshot_quit:
                        self.setVisible(False)
                    else:
                        for widget in self:
                            widget.setVisible(True)

    def mouseMotionEvent(self, p, rel, button, modifiers):
        if super(TestApp, self).mouseMotionEvent(p, rel, button, modifiers):
            return True
        if self.camera.motion(rel):
            return True
        return False

    def mouseButtonEvent(self, p, button, pressed, modifiers):
        if super(TestApp, self).mouseButtonEvent(p, button, pressed, modifiers):
            return True
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.camera.button(p, pressed)
            return True
        if button == glfw.MOUSE_BUTTON_RIGHT and pressed:
            model = np.eye(4)
            projection = self.camera.matrix()

            point = nanogui.unproject((p[0], p[1], 0.0), model, projection, self.size())

            diff = np.array([point[i] - self.camera.position[i] for i in range(3)])
            logger.info(point)
            diff /= np.linalg.norm(diff)
            logger.info(diff)
            logger.info(self.camera.direction)
            logger.info(np.dot(diff, self.camera.direction))
            return True
        return False

    def scrollEvent(self, p, rel):
        if super(TestApp, self).scrollEvent(p, rel):
            return True

        linear = math.sqrt(self.speed / 5.0)
        self.speed = math.pow(max(0.1, min(linear + 0.05 * rel[1], 1)), 2) * 5.0
        return True

    def keyboardEvent(self, key, scancode, action, modifiers):
        if super(TestApp, self).keyboardEvent(key, scancode, action, modifiers):
            return True
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.setVisible(False)
            return True

        if key == glfw.KEY_H and action == glfw.PRESS:
            for widget in self:
                widget.setVisible(not widget.visible())

        if key == glfw.KEY_TAB and action == glfw.PRESS:
            if self.colorCoding.valueButton.pushed():
                self.colorCoding.valueButton.setPushed(False)
                self.colorCoding.bbButton.setPushed(True)
                self.colorCoding.state = ColorCoding.BOUNDINGBOX | (
                    self.colorCoding.state & ColorCoding.SHADING
                )
            elif self.colorCoding.bbButton.pushed():
                self.colorCoding.bbButton.setPushed(False)
                self.colorCoding.state = ColorCoding.NONE | (
                    self.colorCoding.state & ColorCoding.SHADING
                )
            else:
                self.colorCoding.valueButton.setPushed(True)
                self.colorCoding.state = ColorCoding.VALUE | (
                    self.colorCoding.state & ColorCoding.SHADING
                )
            self.__updateColorCoding()
            return True

        if key == glfw.KEY_F and action == glfw.PRESS:
            self.camera.zoomToBox(self.boundingBox[0], self.boundingBox[1])
            return True

        if key == glfw.KEY_W and (action == glfw.PRESS or action == glfw.REPEAT):
            offset = [self.speed * self.camera.direction[i] for i in range(3)]
            self.camera.offsetPosition(offset)
            return True

        if key == glfw.KEY_S and (action == glfw.PRESS or action == glfw.REPEAT):
            offset = [self.speed * -self.camera.direction[i] for i in range(3)]
            self.camera.offsetPosition(offset)
            return True

        if key == glfw.KEY_D and (action == glfw.PRESS or action == glfw.REPEAT):
            right = self.camera.right()
            offset = [self.speed * right[i] for i in range(3)]
            self.camera.offsetPosition(offset)
            return True

        if key == glfw.KEY_A and (action == glfw.PRESS or action == glfw.REPEAT):
            right = self.camera.right()
            offset = [self.speed * -right[i] for i in range(3)]
            self.camera.offsetPosition(offset)
            return True
        return False

    def constructCutplaneWindow(self):
        window = Window(self, "Cutplanes")
        window.setPosition((15, 785))
        layout = GridLayout(Orientation.Horizontal, 2, Alignment.Middle, 15, 5)
        layout.setColAlignment([Alignment.Maximum, Alignment.Fill])
        layout.setSpacing(0, 10)
        window.setLayout(layout)

        Widget(window)  # dummy placeholder

        def keepCB(state):
            self.keepClipping = state

        keepCb = CheckBox(window, "Keep clipping", keepCB)
        keepCb.setTooltip("Clip grids while loading with these values")

        cutplane_indices = {
            "X+": self.shader.addCutplane(
                Cutplane([self.boundingBox[0][0], 0.0, 0.0], [1.0, 0.0, 0.0])
            ),
            "X-": self.shader.addCutplane(
                Cutplane([self.boundingBox[1][0], 0.0, 0.0], [-1.0, 0.0, 0.0])
            ),
            "Y+": self.shader.addCutplane(
                Cutplane([0.0, self.boundingBox[0][1], 0.0], [0.0, 1.0, 0.0])
            ),
            "Y-": self.shader.addCutplane(
                Cutplane([0.0, self.boundingBox[1][1], 0.0], [0.0, -1.0, 0.0])
            ),
            "Z+": self.shader.addCutplane(
                Cutplane([0.0, 0.0, self.boundingBox[0][2]], [0.0, 0.0, 1.0])
            ),
            "Z-": self.shader.addCutplane(
                Cutplane([0.0, 0.0, self.boundingBox[1][2]], [0.0, 0.0, -1.0])
            ),
        }

        def constructCutplaneAxis(label, index, normal):
            Label(window, label)
            panel = Widget(window)
            panel.setLayout(BoxLayout(Orientation.Horizontal, Alignment.Middle, 0, 20))

            slider = Slider(panel)
            slider.setValue(0.0 if "+" in label else 1.0)
            slider.setFixedWidth(80)

            textBox = FloatBox(panel)
            textBox.setFixedSize((60, 20))
            textBox.setValue(0.0 if "+" in label else 100.0)
            textBox.setUnits("%")
            textBox.setFontSize(16)
            textBox.setAlignment(FloatBox.Alignment.Right)
            textBox.setSpinnable(True)
            textBox.setValueIncrement(0.1)
            textBox.setMinValue(0, 100)

            def cb(value):
                textBox.setValue(round(value * 100, 1))

            slider.setCallback(cb)

            def sliderCB(value):
                print("Final slider value: %f" % round(value * 100, 2))

                # update index clipping

                dimensions = [self.boundingBoxIndex[1][index], self.boundingBoxIndex[0][index]]
                if "+" not in label:
                    dimensions = dimensions[::-1]
                indexValue = lerp(
                    dimensions[0], dimensions[1], value if "+" in label else 1.0 - value
                )
                if "+" not in label:
                    self.boundingBoxIndexClipped[1][index] = int(math.ceil(indexValue))
                else:
                    self.boundingBoxIndexClipped[0][index] = int(math.floor(indexValue))

                self.shader.bind()
                position = [0.0, 0.0, 0.0]
                dimensions = [self.boundingBox[1][index], self.boundingBox[0][index]]
                if "+" not in label:
                    dimensions = dimensions[::-1]
                position[index] = lerp(
                    dimensions[0], dimensions[1], value if "+" in label else 1.0 - value
                )
                self.shader.editCutplane(cutplane_indices[label], Cutplane(position, normal))

            slider.setFinalCallback(sliderCB)

            textBox.setCallback(lambda x: sliderCB(x / 100.0) or slider.setValue(x / 100.0))

        for label in sorted(cutplane_indices):
            index = cutplane_indices[label]
            constructCutplaneAxis(
                label, ord(label[0]) - ord("X"), self.shader.cutplanes[index].normal
            )

    #######################
    #   OpenVDB Section   #
    #######################

    def loadVDBFile(self, filename):
        if not self.vdb.grids:
            self.vdb.grids, self.vdb.metadata = vdb.readAll(filename)

            self.vdb.window = Window(self, "OpenVDB File")
            self.vdb.window.setPosition((15, 300))
            self.vdb.window.setFixedSize((250, 470))
            self.vdb.window.setLayout(BoxLayout(Orientation.Vertical))
            vscroll = VScrollPanel(self.vdb.window)
            vscroll.setFixedSize((250, 440))
            self.vdb.panel = Widget(vscroll)
            self.vdb.panel.setLayout(GroupLayout())

            metadata = []

            for key in self.vdb.metadata.keys():
                metadata.append("{} = {}".format(key, self.vdb.metadata[key]))

            if metadata:
                Label(self.vdb.panel, "MetaData", "sans-bold")
                Label(self.vdb.panel, "\n".join(metadata))

            for idx, grid in enumerate(self.vdb.grids):
                b = Button(self.vdb.panel, "Name: {}".format(grid.name))
                b.setFontSize(16)
                b.setCallback(lambda idx=idx: self.showGrid(idx))

                metadata = []

                for key in grid.metadata.keys():
                    metadata.append("{} = {}".format(key, grid.metadata[key]))

                if metadata:
                    b.setTooltip("\n".join(metadata) + "\nBackground = " + str(grid.background))

            self.performLayout()

    def showGrid(self, gridname, focus=False):
        if not self.vdb.grids and self.filename.endswith(".vdb"):
            self.loadVDBFile(self.filename)
        elif self.filename.endswith("_discrete_volume.gz"):
            self.loadDiscreteVolume(self.filename)
            return
        else:
            logger.error(
                "Unknown File Type: {}. We can load .vdb and _discrete_volume.gz".format(
                    self.filename
                )
            )

        try:
            gridindex = int(gridname)

            if not 0 <= gridindex < len(self.vdb.grids):
                logger.error(
                    "Grid index {} is out of range [0, {}]".format(
                        gridindex, len(self.vdb.grids) - 1
                    )
                )
                return

            grid = self.vdb.grids[gridindex]
        except ValueError:
            try:
                gridindex, grid = next(
                    (idx, grid)
                    for idx, grid in enumerate(self.vdb.grids)
                    if grid.name == gridname
                )
            except StopIteration:
                logger.error("Could not find grid with name {}".format(gridname))
                logger.info(
                    "Choices are: [{}]".format(
                        ",".join(["'" + grid.name + "'" for grid in self.vdb.grids])
                    )
                )
                return

        if self.shader.attribVersion("position") == gridindex:
            # already uploaded this grid
            return

        if self.keepClipping:
            grid = grid.deepCopy()
            logger.info("Clipping Grid to {}".format(self.boundingBoxIndexClipped))
            # mark 6 cubes around as inactive and prune meanwhile
            for side in range(2):  # lower / upper
                for dim in range(3):  # xyz
                    box = copy.deepcopy(self.boundingBoxIndex)
                    box[not side][dim] = self.boundingBoxIndexClipped[side][dim]
                    grid.fill(*box, value=grid.background, active=False)
                    grid.pruneInactive()

        visualize_correspondence = (
            grid.name.lower().find("nearestsurfacepoints") is not -1
            and grid.name.lower().find("nearestsurfacepointsgt") is -1
        )
        visualize_normals = grid.name.lower().find("normalgrid") is not -1
        self.__toggleShader(voxelShader=not (visualize_correspondence or visualize_normals))

        # reserve space for 24 vectors per Voxel
        num_triangles = grid.activeVoxelCount()  # * 6 * 2
        if visualize_correspondence or visualize_normals:
            num_triangles *= 2
        positions = np.zeros(shape=(3, num_triangles), dtype=np.float32)
        colors = np.zeros(shape=(3, num_triangles), dtype=np.float32)
        # indices = np.zeros(shape=(3, num_triangles), dtype=np.int32)

        logger.info(
            "Loading Grid '{}'[{}] with {} voxels".format(grid.name, gridindex, num_triangles)
        )

        if num_triangles == 0:
            self.shader.bind()
            self.shader.freeAttrib("position")
            self.shader.freeAttrib("color")
            self.num_triangles = num_triangles
            return

        voxelSize = [1, 1, 1]

        progress_length = 40
        timer = get_timer()
        start = timer()
        progress_update = min(0.1, 1000000 / (2 * num_triangles))
        progress = progress_update

        if visualize_correspondence:
            correspondenceLevel = int(
                grid.name[
                    grid.name.lower().find("nearestsurfacepoints") + len("nearestsurfacepoints") :
                ]
            )
            weightGridIndex, weightGridChannel = divmod(correspondenceLevel, 3)
            weightGridname = "correspondenceWeights{}".format(weightGridIndex)
            try:
                _, weightgrid = next(
                    (idx, grid)
                    for idx, grid in enumerate(self.vdb.grids)
                    if grid.name == weightGridname
                )
                wAccessor = weightgrid.getConstAccessor()
            except StopIteration:
                weightgrid = None
                wAccessor = None

        counter = 0
        for item in grid.citerOnValues():
            if item.count == 1:  # voxel value
                positions[:, counter] = (item.min[0], item.min[1], item.min[2])

                if type(item.value) == tuple and len(item.value) == 3:
                    colorValue = item.value
                else:
                    colorValue = [item.value] * 3

                colors[:, counter] = colorValue

                if visualize_correspondence:
                    positions[:, counter + 1] = (
                        colorValue if (item.min != colorValue) else np.array(colorValue) + 0.1
                    )
                    if wAccessor:
                        colorValue = wAccessor.getValue(item.min)[weightGridChannel]
                        colors[:, counter] = [colorValue] * 3
                        colors[:, counter + 1] = [colorValue] * 3
                    else:
                        colors[:, counter + 1] = colorValue
                    counter += 2
                elif visualize_normals:
                    colors[:, counter] *= 0.01
                    positions[:, counter + 1] = (
                        item.min[0] + colorValue[0],
                        item.min[1] + colorValue[1],
                        item.min[2] + colorValue[2],
                    )
                    colors[:, counter + 1] = colorValue
                    counter += 2
                else:
                    counter += 1
                if counter / num_triangles >= progress:
                    length = int(progress_length * progress)
                    elapsed = timer() - start

                    ETA = (elapsed * num_triangles / counter) * (1.0 - counter / num_triangles)

                    format_string = (
                        "{: >5.1f}% [{: <" + str(progress_length) + "}] ETA: {:>3.0f}m {:>2.0f}s "
                    )
                    logger.info(
                        format_string.format(
                            progress * 100, "=" * length, int(ETA / 60), ETA % 60
                        )
                    )
                    progress += progress_update

        self.shader.bind()
        self.shader.freeAttrib("position")
        self.shader.uploadAttrib("position", positions, gridindex)

        self.shader.freeAttrib("color")
        self.shader.uploadAttrib("color", colors, gridindex)

        if visualize_correspondence and weightgrid:
            weightgridCopy = weightgrid.deepCopy()
            logger.info("Clipping Grid to {}".format(self.boundingBoxIndexClipped))
            # mark 6 cubes around as inactive and prune meanwhile
            for side in range(2):  # lower / upper
                for dim in range(3):  # xyz
                    box = copy.deepcopy(self.boundingBoxIndex)
                    box[not side][dim] = self.boundingBoxIndexClipped[side][dim]
                    weightgridCopy.fill(*box, value=weightgrid.background, active=False)
                    weightgridCopy.pruneInactive()
            minMax = weightgridCopy.evalMinMax()
            minMax = (minMax[0][weightGridChannel], minMax[1][weightGridChannel])
        else:
            minMax = grid.evalMinMax()
        if type(minMax[0]) != tuple:
            minMax = ([minMax[0]] * 3, [minMax[1]] * 3)

        self.__setMinMax(minMax[0], minMax[1])

        self.num_triangles = num_triangles
        # HACK to get the matrix from openvdb transform
        model = eval(
            "["
            + ",".join([str.strip() for str in grid.transform.info().split("\n")[2 : 2 + 4]])
            + "]"
        )
        self.model_matrix = [
            [float(model[y][x]) for x in range(len(model[y]))] for y in range(len(model))
        ]

        if not self.keepClipping:
            bbMin, bbMax = grid.evalActiveVoxelBoundingBox()

            self.boundingBoxIndex[0] = list(bbMin)
            self.boundingBoxIndex[1] = [dim + 1 for dim in list(bbMax)]
            self.boundingBoxIndexClipped[0] = self.boundingBoxIndex[0]
            self.boundingBoxIndexClipped[1] = self.boundingBoxIndex[1]
            self.boundingBox[0] = np.dot(self.model_matrix, bbMin + (1,))[:3]
            self.boundingBox[1] = np.dot(self.model_matrix, bbMax + (1,))[:3]

        if focus:
            self.camera.zoomToBox(self.boundingBox[0], self.boundingBox[1])

    def loadDiscreteVolume(self, filename):
        grid = LabelGrid(filename)

        def loadPositions():
            positions = np.zeros(shape=(3,) + grid.data.shape, dtype=np.float32)
            x = np.linspace(0, grid.res_x - 1, grid.res_x)
            y = np.linspace(0, grid.res_y - 1, grid.res_y)
            z = np.linspace(0, grid.res_z - 1, grid.res_z)
            meshgrid = np.meshgrid(x, y, z, indexing="ij")

            positions[0, :, :, :] = meshgrid[0].transpose()
            positions[1, :, :, :] = meshgrid[1].transpose()
            positions[2, :, :, :] = meshgrid[2].transpose()
            positions = positions[:, grid.data <= 4]  # filter for Transparency + Air
            positions = positions.reshape(3, -1)

            self.shader.bind()
            self.shader.freeAttrib("position")
            self.shader.uploadAttrib("position", positions, 0)
            self.num_triangles = positions.shape[1]

        def loadColors():
            colors = np.zeros(shape=grid.data.shape + (3,), dtype=np.float32)
            # colors = np.random.rand(3, *grid.data.shape)
            colors[grid.data == 0] = (0.0, 1.0, 1.0)  # Cyan
            colors[grid.data == 1] = (1.0, 0.0, 1.0)  # Magenta
            colors[grid.data == 2] = (1.0, 1.0, 0.0)  # Yellow
            colors[grid.data == 3] = (0.0, 0.0, 0.0)  # Black
            colors[grid.data == 4] = (1.0, 1.0, 1.0)  # White
            colors = colors[grid.data <= 4, :]  # filter for Transparency + Air
            colors = np.moveaxis(colors, -1, 0)  # move RGB to front
            colors = colors.reshape(3, -1)

            self.shader.bind()
            self.shader.freeAttrib("color")
            self.shader.uploadAttrib("color", colors, 0)

        loadPositions()
        gc.collect()
        loadColors()
        gc.collect()

        extend = grid.bbox_max - grid.bbox_min
        self.model_matrix = np.eye(4) * np.array(
            [extend[0] / grid.res_x, extend[1] / grid.res_y, extend[2] / grid.res_z, 1.0]
        )
        self.model_matrix[:3, 3] = grid.bbox_min

        self.boundingBoxIndex[0] = [0, 0, 0]
        self.boundingBoxIndex[1] = [grid.res_x, grid.res_y, grid.res_z]
        self.boundingBoxIndexClipped[0] = self.boundingBoxIndex[0]
        self.boundingBoxIndexClipped[1] = self.boundingBoxIndex[1]
        self.boundingBox[0] = grid.bbox_min
        self.boundingBox[1] = grid.bbox_max

    def __setMinMax(self, minValue, maxValue):
        self.colorCoding.min = (float(minValue[0]), float(minValue[1]), float(minValue[2]))
        self.colorCoding.max = (float(maxValue[0]), float(maxValue[1]), float(maxValue[2]))
        self.colorCoding.minTextbox.setValue(self.colorCoding.min[0])
        self.colorCoding.maxTextbox.setValue(self.colorCoding.max[0])
        self.__updateColorCoding()  # update if needed

    def __updateColorCoding(self):
        self.shader.bind()
        self.shader.setUniform("u_colorCode", int(self.colorCoding.state))
        self.shader.setUniform("u_valueRangeMin", self.colorCoding.min)
        self.shader.setUniform("u_valueRangeMax", self.colorCoding.max)

    def __toggleShader(self, voxelShader=None):
        if voxelShader is None:
            voxelShader = type(self.shader) is CorrespondenceShader

        oldShader = self.shader
        if voxelShader is True and type(self.shader) is not VoxelShader:
            self.shader = VoxelShader("Voxel shader")
        elif voxelShader is False and type(self.shader) is not CorrespondenceShader:
            self.shader = CorrespondenceShader("Correspondence shader")
        else:
            return

        print(type(oldShader), type(self.shader))
        self.shader.lights = oldShader.lights
        self.shader.cutplanes = oldShader.cutplanes
        [print(light) for light in self.shader.lights]
        self.shader.bind()
        self.shader.update()
        self.__updateColorCoding()


def get_argument_parser():
    parser = argparse.ArgumentParser(description="View openvdb grids interactively.")
    parser.add_argument("filename", type=str, help="the file to display")
    parser.add_argument(
        "gridname",
        type=str,
        help="the name of the grid (or index) to display",
        nargs="?",
        default="0",
    )
    parser.add_argument(
        "-p",
        "--position",
        type=str,
        help="the camera position, multiple allowed",
        default=argparse.SUPPRESS,
        action="append",
    )
    parser.add_argument(
        "-d",
        "--direction",
        type=str,
        help="the camera direction, multiple allowed",
        default=argparse.SUPPRESS,
        action="append",
    )
    parser.add_argument(
        "-c",
        "--crop",
        type=int,
        help="crop <x,y,w,h> in pixel coordinates",
        nargs=4,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-s",
        "--screenshot",
        type=str,
        help="save a screenshot under this path",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-q",
        "--quit",
        help="close the application after screenshots have been taken",
        default=False,
        action="store_true",
    )
    return parser


def make_vdb_screenshots(
    filename: str, gridname: str, views: List, background: int, output_base_filename: str
) -> List[str]:
    parser = get_argument_parser()
    args = parser.parse_args(
        [
            filename,
            gridname,
            "-p",
            str([v["position"] for v in views]),
            "-d",
            str([v["direction"] for v in views]),
            "--screenshot",
            output_base_filename,
            "--quit",
        ]
    )

    nanogui.init()
    test = TestApp(args, background)
    test.drawAll()
    test.setVisible(True)
    nanogui.mainloop()
    del test
    gc.collect()
    nanogui.shutdown()

    filename_base, filename_ext = os.path.splitext(output_base_filename)
    return [filename_base + "_" + str(idx) + filename_ext for idx in range(len(views))]


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    nanogui.init()
    test = TestApp(args)
    test.drawAll()
    test.setVisible(True)
    nanogui.mainloop()
    del test
    gc.collect()
    nanogui.shutdown()


if __name__ == "__main__":
    main()
