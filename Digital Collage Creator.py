import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageEnhance
from PIL import ImageColor
import math
import os
from matplotlib.path import Path as PlotPath
import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import Counter
import random
import uuid
from pathlib import Path
import resources
import json
import base64
from io import BytesIO
import shutil

# Set the path to the temporary subdirectory used to store
# original and edited layer images.
project_path = Path("project/")

# Action Manager


class AMTokens():
    # Defines a list of action tokens used to describe and identify
    # the function of an action.
    layer_added_token = "LYRADD"
    layer_deleted_token = "LYRDLT"
    layer_moved_token = "LYRMOV"
    layer_rotated_token = "LYRROT"
    layer_scaled_token = "LYRSCL"
    layer_cut_token = "LYRCUT"
    layer_cropped_token = "LYRCRP"
    blur_token = "LYRBLR"
    baw_token = "LYRBAW"
    brightness_token = "LYRBRI"
    sharpness_token = "LYRSHRP"
    contrast_token = "LYRCON"
    rgb_token = "LYRRGB"
    layer_moved_down_token = "LYRDWN"
    layer_moved_up_token = "LYRUP"
    active_layer_change_token = "ACTLYR"
    layer_visible_change_token = "LYRVIS"
    layer_randomise_start_token = "STRTRDM"
    layer_randomise_end_token = "ENDRDM"


class ActionManager():
    # Stores action's performed by the user onto a list.
    # The list is used as a stack data structure.
    # LIFO (Last in, First out). To undo an action an action is popped
    # from the stack, the token attached to the action is identified
    # in order to pass the action to the relevant undo function.
    # Once the action has been undone the action is placed into
    # the removed actions stack. To redo an action, actions are
    # popped from the removed actions stack and passed to the
    # relavant redo function. A user sanctioned action will
    # empty the removed actions stack.

    action_stack = []
    removed_actions = []

    redo_flag = False
    currently_redoing = False
    currently_undoing = False
    undoing_randomise = False
    redoing_randomise = False

    def action():
        # Resets the removed actions stack if an action is
        # executed by the user.
        if ActionManager.redo_flag == False:
            ActionManager.emptyStack()
        ActionManager.redo_flag = False

    # Each reversible action calls it's related action manager function
    # when executed. The function pushes the action onto the action stack
    # as a list. The first element contains the identifying action token,
    # subsequent elements are parameters required for the undoing and
    # redoing of the action.

    def layerAdded(layer):
        ActionManager.action()
        ActionManager.action_stack.append([AMTokens.layer_added_token, layer])

    def layerDeleted(layer):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_deleted_token, layer])

    def layerMoved(layer, orig_x, orig_y, new_x, new_y):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_moved_token, layer, orig_x, orig_y, new_x, new_y])

    def layerRotated(layer, orig_angle, new_angle):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_rotated_token, layer, orig_angle, new_angle])

    def layerScaled(layer, orig_scale, new_scale):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_scaled_token, layer, orig_scale, new_scale])

    def layerCut(layer, orig_image, new_image):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_cut_token, layer, orig_image, new_image])

    def layerCropped(layer, orig_image, new_image, coordinates):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_cropped_token, layer, orig_image, new_image, coordinates])

    def blurChanged(layer, orig_blur, new_blur):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.blur_token, layer, orig_blur, new_blur])

    def bawChanged(layer, orig_baw, new_baw):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.baw_token, layer, orig_baw, new_baw])

    def brightnessChanged(layer, orig_brightness, new_brightness):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.brightness_token, layer, orig_brightness, new_brightness])

    def sharpnessChanged(layer, orig_sharpness, new_sharpness):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.sharpness_token, layer, orig_sharpness, new_sharpness])

    def contrastChanged(layer, orig_contrast, new_contrast):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.contrast_token, layer, orig_contrast, new_contrast])

    def rgbChanged(layer, orig_rgb, new_rgb):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.rgb_token, layer, orig_rgb, new_rgb])

    def layerMovedDown(layer):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_moved_down_token, layer])

    def layerMovedUp(layer):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_moved_up_token, layer])

    def activeLayerChanged(orig_active, new_active):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.active_layer_change_token, orig_active, new_active])

    def layerVisibleChanged(layer, orig_visible, new_visible):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_visible_change_token, layer, orig_visible, new_visible])

    def layerRandomiseStart(layer):
        ActionManager.action()
        ActionManager.action_stack.append(
            [AMTokens.layer_randomise_start_token])

    def layerRandomiseEnd(layer):
        ActionManager.action()
        ActionManager.action_stack.append([AMTokens.layer_randomise_end_token])

    # An undo and redo function is defined for each category
    # of action. The function receives an action as a parameter
    # and uses the values in the action list to reverse or
    # reapply the effect of the action.

    # Functions to undo and redo the randomisation of a layer's
    # properties. Randomising a layer is one action performed by
    # the user that can create multiple actions perfomed by the
    # program (changes of multiple properties). Start and end tokens
    # are used to identify the set of actions caused by the
    # user action.
    def undoRandomiseLayer(action):
        ActionManager.undoing_randomise = True
        undo = True

        # Add the "end of randomisation" token to the removed actions stack.
        ActionManager.removed_actions.append(action)

        # Calls the undo action function on actions in the action
        # stack until the "start of randomisation" token is reached.
        while (undo):
            action_to_undo = ActionManager.action_stack.pop()
            if action_to_undo[0] == AMTokens.layer_randomise_start_token:
                undo = False
                ActionManager.removed_actions.append(action_to_undo)
            ActionManager.undo(action_to_undo)
        ActionManager.undoing_randomise = False

    def redoRandomiseLayer(action):
        ActionManager.redoing_randomise = True
        redo = True

        # Add the "start of randomisation" token to the action stack.
        ActionManager.action_stack.append(action)

        # Calls the redo action function on actions in the removed
        # action stack until the "end of randomisation" token is reached.
        while (redo):
            action_to_redo = ActionManager.removed_actions.pop()
            if action_to_redo[0] == AMTokens.layer_randomise_end_token:
                redo = False
                ActionManager.action_stack.append(action_to_redo)
            ActionManager.redo(action_to_redo)
        ActionManager.redoing_randomise = False

    # Functions to undo and redo changes made to a layer's visibility.
    def undoLayerVisibleChange(action):
        layer = action[1]
        orig_visible = action[2]
        if orig_visible:
            layer.turnVisibleOn()
        else:
            layer.turnVisibleOff()
        ActionManager.removed_actions.append(action)

    def redoLayerVisibleChange(action):
        layer = action[1]
        new_visible = action[3]
        if new_visible:
            layer.turnVisibleOn()
        else:
            layer.turnVisibleOff()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo actions that change
    # which layer is active.
    def undoActiveLayerChange(action):
        orig_active = action[1]
        new_active = action[2]
        if orig_active is not None:
            orig_active.getLayerWidget().layerActiveOn()
        else:
            new_active.getLayerWidget().layerActiveOff()
        ActionManager.removed_actions.append(action)

    def redoActiveLayerChange(action):
        orig_active = action[1]
        new_active = action[2]
        if new_active is not None:
            new_active.getLayerWidget().layerActiveOn()
        else:
            orig_active.getLayerWidget().layerActiveOff()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo moving a layer down.
    def undoLayerDown(action):
        layer = action[1]
        LayerManager.moveLayerUp(layer)
        ActionManager.removed_actions.append(action)

    def redoLayerDown(action):
        layer = action[1]
        LayerManager.moveLayerDown(layer)
        ActionManager.action_stack.append(action)

    # Functions to undo and redo moving a layer up.
    def undoLayerUp(action):
        layer = action[1]
        LayerManager.moveLayerDown(layer)
        ActionManager.removed_actions.append(action)

    def redoLayerUp(action):
        layer = action[1]
        LayerManager.moveLayerUp(layer)
        ActionManager.action_stack.append(action)

    # Functions to undo and redo changes made to a layer's
    # red, green and blue intensity values.
    def undoRGBChange(action):
        layer = action[1]
        orig_rgb = action[2]
        layer.setRGB(orig_rgb[0], orig_rgb[1], orig_rgb[2])
        layer.applyAlterations()
        ActionManager.removed_actions.append(action)

    def redoRGBChange(action):
        layer = action[1]
        new_rgb = action[3]
        layer.setRGB(new_rgb[0], new_rgb[1], new_rgb[2])
        layer.applyAlterations()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo the application/removal
    # of the black and white filter onto a layer.
    def undoBawChange(action):
        layer = action[1]
        orig_baw = action[2]
        layer.setBW(orig_baw)
        layer.applyAlterations()
        ActionManager.removed_actions.append(action)

    def redoBawChange(action):
        layer = action[1]
        new_baw = action[3]
        layer.setBW(new_baw)
        layer.applyAlterations()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo the changing of a layer's
    # brightness property value.
    def undoBrightnessChange(action):
        layer = action[1]
        orig_brightness = action[2]
        layer.setBrightness(orig_brightness)
        layer.applyAlterations()
        ActionManager.removed_actions.append(action)

    def redoBrightnessChange(action):
        layer = action[1]
        new_brightness = action[3]
        layer.setBrightness(new_brightness)
        layer.applyAlterations()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo the changing of a layer's
    # sharpness property value.
    def undoSharpnessChange(action):
        layer = action[1]
        orig_sharpness = action[2]
        layer.setSharpness(orig_sharpness)
        layer.applyAlterations()
        ActionManager.removed_actions.append(action)

    def redoSharpnessChange(action):
        layer = action[1]
        new_sharpness = action[3]
        layer.setSharpness(new_sharpness)
        layer.applyAlterations()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo the changing of a layer's
    # contrast property value.
    def undoContrastChange(action):
        layer = action[1]
        orig_contrast = action[2]
        layer.setContrast(orig_contrast)
        layer.applyAlterations()
        ActionManager.removed_actions.append(action)

    def redoContrastChange(action):
        layer = action[1]
        new_contrast = action[3]
        layer.setContrast(new_contrast)
        layer.applyAlterations()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo the application/removal
    # of the blur filter on a layer.
    def undoBlurChange(action):
        layer = action[1]
        orig_blur = action[2]
        layer.setBlur(orig_blur)
        layer.applyAlterations()
        ActionManager.removed_actions.append(action)

    def redoBlurChange(action):
        layer = action[1]
        new_blur = action[3]
        layer.setBlur(new_blur)
        layer.applyAlterations()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo the cropping of an image.
    def undoLayerCrop(action):
        layer = action[1]
        orig_image = action[2]
        coordinates = action[4]
        orig_x, orig_y = coordinates[0], coordinates[1]
        cropped_image_name = layer.cropped_image_name
        orig_image.save(str(project_path / cropped_image_name))
        layer.setXY(orig_x, orig_y)
        setOriginToCenter(layer.getLayerItem())
        layer.applyAlterations()
        ActionManager.removed_actions.append(action)

    def redoLayerCrop(action):
        layer = action[1]
        new_image = action[3]
        coordinates = action[4]
        new_x, new_y = coordinates[2], coordinates[3]
        cropped_image_name = layer.cropped_image_name
        new_image.save(str(project_path / cropped_image_name))
        layer.setXY(new_x, new_y)
        layer.applyAlterations()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo the cutting of an image.
    def undoLayerCut(action):
        layer = action[1]
        orig_image = action[2]
        cropped_image_name = layer.cropped_image_name
        orig_image.save(str(project_path / cropped_image_name))
        layer.applyAlterations()
        ActionManager.removed_actions.append(action)

    def redoLayerCut(action):
        layer = action[1]
        new_image = action[3]
        cropped_image_name = layer.cropped_image_name
        new_image.save(str(project_path / cropped_image_name))
        layer.applyAlterations()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo changes made to a layer's size.
    def undoLayerScaled(action):
        layer = action[1]
        orig_scale = action[2]
        layer.getLayerItem().setScale(orig_scale)
        layer.getScaleItem().positionIcon()
        ActionManager.removed_actions.append(action)

    def redoLayerScaled(action):
        layer = action[1]
        new_scale = action[3]
        layer.getLayerItem().setScale(new_scale)
        layer.getScaleItem().positionIcon()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo changes made to a layer's rotation.
    def undoLayerRotated(action):
        layer = action[1]
        orig_angle = action[2]
        layer.getLayerItem().setRotation(orig_angle)
        layer.getRotateItem().positionIcon()
        ActionManager.removed_actions.append(action)

    def redoLayerRotated(action):
        layer = action[1]
        new_angle = action[3]
        layer.getLayerItem().setRotation(new_angle)
        layer.getRotateItem().positionIcon()
        ActionManager.action_stack.append(action)

    # Functions to undo and redo changes made to a layer's position.
    def undoLayerMoved(action):
        layer = action[1]
        x, y = action[2], action[3]
        layer.setPos(x, y)
        ActionManager.removed_actions.append(action)

    def redoLayerMoved(action):
        layer = action[1]
        x, y = action[4], action[5]
        layer.setPos(x, y)
        ActionManager.action_stack.append(action)

    # Functions to undo and redo the adding of a layer.
    def undoLayerAdded(action):
        mw.deleteLayer(action[1])
        ActionManager.removed_actions.append(action)

    def redoLayerAdded(action):
        LayerManager.undoDeleteLayer(action[1])
        ActionManager.action_stack.append(action)

    # Functions to undo and redo the deletion of a layer.
    def undoLayerDeleted(action):
        LayerManager.undoDeleteLayer(action[1])
        ActionManager.removed_actions.append(action)

    def redoLayerDeleted(action):
        mw.deleteLayer(action[1])
        ActionManager.action_stack.append(action)

    def undoClick():
        # undoClick() is called when the user clicks the Undo button.
        # The function will do nothing if the program is currently
        # undoing an action.
        if ActionManager.currently_undoing or ActionManager.undoing_randomise:
            return

        if ActionManager.action_stack:
            # If the stack contains actions to undo. The most recent
            # action is passed into undo()
            action_to_undo = ActionManager.action_stack.pop()
            ActionManager.undo(action_to_undo)

            # Display feedback to the user.
            mw.status_bar.showMessage("Action undone...", 2000)
        else:
            # The stack contains no actions.
            mw.status_bar.showMessage("No actions to undo...", 2000)

    def undo(action):
        # Uses the action's token to identify which undo function
        # the action should be passed to.
        ActionManager.currently_undoing = True
        action_to_undo = action
        if (action_to_undo[0] == AMTokens.layer_added_token):
            ActionManager.undoLayerAdded(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_deleted_token):
            ActionManager.undoLayerDeleted(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_moved_token):
            ActionManager.undoLayerMoved(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_rotated_token):
            ActionManager.undoLayerRotated(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_scaled_token):
            ActionManager.undoLayerScaled(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_cut_token):
            ActionManager.undoLayerCut(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_cropped_token):
            ActionManager.undoLayerCrop(action_to_undo)
        elif (action_to_undo[0] == AMTokens.blur_token):
            ActionManager.undoBlurChange(action_to_undo)
        elif (action_to_undo[0] == AMTokens.baw_token):
            ActionManager.undoBawChange(action_to_undo)
        elif (action_to_undo[0] == AMTokens.brightness_token):
            ActionManager.undoBrightnessChange(action_to_undo)
        elif (action_to_undo[0] == AMTokens.contrast_token):
            ActionManager.undoContrastChange(action_to_undo)
        elif (action_to_undo[0] == AMTokens.sharpness_token):
            ActionManager.undoSharpnessChange(action_to_undo)
        elif (action_to_undo[0] == AMTokens.rgb_token):
            ActionManager.undoRGBChange(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_moved_down_token):
            ActionManager.undoLayerDown(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_moved_up_token):
            ActionManager.undoLayerUp(action_to_undo)
        elif (action_to_undo[0] == AMTokens.active_layer_change_token):
            ActionManager.undoActiveLayerChange(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_visible_change_token):
            ActionManager.undoLayerVisibleChange(action_to_undo)
        elif (action_to_undo[0] == AMTokens.layer_randomise_end_token):
            ActionManager.undoRandomiseLayer(action_to_undo)
        ActionManager.currently_undoing = False

    def redoClick():
        # redoClick() is called when the user clicks the Redo button.
        # The function will do nothing if the program is currently
        # in the process of redoing an action.
        if ActionManager.currently_redoing or ActionManager.redoing_randomise:
            return

        if ActionManager.removed_actions:
            # If the stack contains actions to redo then the most
            # recently added action is popped and passed to the redo()
            # function.
            action_to_redo = ActionManager.removed_actions.pop()
            ActionManager.redo(action_to_redo)

            # Feedback is displayed.
            mw.status_bar.showMessage("Action redone...")
        else:
            # The stack contains no actions to redo.
            mw.status_bar.showMessage("No actions to redo...")

    def redo(action):
        # Uses the action's token to identify which redo function
        # the action should be passed to.
        ActionManager.redo_flag = True
        ActionManager.currently_redoing = True
        action_to_redo = action
        if (action_to_redo[0] == AMTokens.layer_added_token):
            ActionManager.redoLayerAdded(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_deleted_token):
            ActionManager.redoLayerDeleted(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_moved_token):
            ActionManager.redoLayerMoved(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_rotated_token):
            ActionManager.redoLayerRotated(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_scaled_token):
            ActionManager.redoLayerScaled(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_cut_token):
            ActionManager.redoLayerCut(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_cropped_token):
            ActionManager.redoLayerCrop(action_to_redo)
        elif (action_to_redo[0] == AMTokens.blur_token):
            ActionManager.redoBlurChange(action_to_redo)
        elif (action_to_redo[0] == AMTokens.baw_token):
            ActionManager.redoBawChange(action_to_redo)
        elif (action_to_redo[0] == AMTokens.brightness_token):
            ActionManager.redoBrightnessChange(action_to_redo)
        elif (action_to_redo[0] == AMTokens.contrast_token):
            ActionManager.redoContrastChange(action_to_redo)
        elif (action_to_redo[0] == AMTokens.sharpness_token):
            ActionManager.redoSharpnessChange(action_to_redo)
        elif (action_to_redo[0] == AMTokens.rgb_token):
            ActionManager.redoRGBChange(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_moved_down_token):
            ActionManager.redoLayerDown(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_moved_up_token):
            ActionManager.redoLayerUp(action_to_redo)
        elif (action_to_redo[0] == AMTokens.active_layer_change_token):
            ActionManager.redoActiveLayerChange(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_visible_change_token):
            ActionManager.redoLayerVisibleChange(action_to_redo)
        elif (action_to_redo[0] == AMTokens.layer_randomise_start_token):
            ActionManager.redoRandomiseLayer(action_to_redo)
        ActionManager.currently_redoing = False

    def emptyStack():
        # Removed all actions from the removed action stack.
        ActionManager.removed_actions = []


class ImagePoint:

    def __init__(self, x, y):
        # Stores the point's coordinates in the image (x, y)
        self.x = x
        self.y = y

        # Creates the item used as a marker to add to the canvas
        point_pixmap = qtg.QPixmap("pixel_point.png")
        self.point_item = qtw.QGraphicsPixmapItem(point_pixmap)

        # Position the point on the canvas
        self.positionPoint()

    def positionPoint(self):
        # Positions the point marker image on the canvas such that
        # the center of the marker image is the (x, y) coordinates
        # of the plotted point.
        bounding_rect = self.point_item.boundingRect()
        self.point_item.setPos(self.x - (bounding_rect.width()/2),
                               self.y - (bounding_rect.height()/2))


class LineManager:
    # Manages the addition and deletion of lines between plotted points
    # in the cutout window when the user is plotting a 2D path over an image.

    def __init__(self, cw):
        self.cw = cw
        self.lines = []
        self.removed = []

    def removeLine(self):
        # Removes the most recently added line.
        if self.lines:
            # If the list of lines is not empty the most recently added
            # will be popped from the list and added to the removed lines
            # list.
            line_to_remove = self.lines.pop()
            self.removed.append(line_to_remove)

            # Deletes the line from the cutout canvas.
            self.cw.graphics_scene.removeItem(line_to_remove)

    def addLine(self, x0, y0, x1, y1):
        # Takes two coordinates (x0, y0) and (x1, y1) and draws a line
        # on the canvas between the two. Each line is stored in the
        # object's list of lines.
        new_line = self.cw.graphics_scene.addLine(x0, y0, x1, y1, self.cw.pen)
        self.lines.append(new_line)

    def emptyStack(self):
        # Resets the list of removed lines.
        self.removed = []

    def removeAllLines(self):
        while self.lines:
            self.removeLine()


class PointManager():
    # Manages the points plotted by the user when drawing the path
    # over an image in the cutout window.

    def __init__(self, cw):
        self.cw = cw
        self.points = []
        self.removed = []
        self.redo_flag = False

    def setLineManager(self):
        self.line_manager = self.cw.getLineManager()

    def addPoint(self, x, y):
        if self.redo_flag is False:
            # If the point is added by the user (i.e. the user clicks on the
            # image and not by the program (i.e. a point being re-added)
            # then the stack of removed points is emptied.
            self.emptyStack()

        self.redo_flag = False

        # Create and add the point marker image to the canvas
        self.new_point = ImagePoint(x, y)
        self.new_point_item = self.cw.graphics_scene.addItem(
            self.new_point.point_item)

        if self.points:
            # If the point being added is not the first point added
            # then a line is drawn between the new point and the
            # most recent point before it.
            self.previous_point = self.points[-1]
            self.line_manager.addLine(
                self.previous_point.x, self.previous_point.y, x, y)

        # Store the added point in the list of points.
        self.points.append(self.new_point)

    def emptyStack(self):
        # Empty the list of removed points.
        self.removed = []
        self.line_manager.emptyStack()

    def undo(self):
        if self.points:
            # Take the most recently added point and push it onto
            # the stack of removed points.
            self.point_to_remove = self.points.pop()
            self.removed.append(self.point_to_remove)

            # Remove the point marker image from the canvas and
            # call line manager to remove the most recently drawn line.
            self.cw.graphics_scene.removeItem(self.point_to_remove.point_item)
            self.line_manager.removeLine()

            mw.status_bar.showMessage("Point undone...", 4000)
        else:
            # The list of points is empty.
            mw.status_bar.showMessage("No points to undo...", 4000)

    def redo(self):
        if self.removed:
            # Take the most recently deleted point's coordinates (x,y)
            # and add a new point at (x,y)
            self.point_to_add = self.removed.pop()
            self.redo_flag = True
            self.addPoint(self.point_to_add.x, self.point_to_add.y)
            mw.status_bar.showMessage("Point redone...", 4000)
        else:
            # The list of removed points is empty.
            mw.status_bar.showMessage("No points to redo...", 4000)

    def joinMask(self):
        if self.points:
            # Check if the first and last points have the same coordinates
            if ((self.points[0].x == self.points[-1].x) and (self.points[0].y == self.points[-1].y)):
                # Path is already joined
                mw.status_bar.showMessage("Path is already joined...", 4000)
                return

            # Adds a new point to the path at the same coordinates as the
            # starting point.
            self.first_point = self.points[0]
            self.addPoint(self.first_point.x, self.first_point.y)
            mw.status_bar.showMessage("Mask joined...", 4000)

    def removeAllPoints(self):
        self.removed = []
        while self.points:
            self.point_to_remove = self.points.pop()
            self.cw.graphics_scene.removeItem(self.point_to_remove.point_item)

    def maskClicked(self, blurAmount):
        # Called when the user clicks the "mask" option button.
        # Calls the image masking functions only if the user has
        # plotted points in the path.
        if self.points:
            self.maskImageWithBlur(blurAmount)

    def maskImageWithBlur(self, blurAmount):
        cropped_image_name = self.cw.cropped_image_name
        img = Image.open(str(project_path / cropped_image_name))
        pre_cut_image = img.copy()
        pre_cut_image.load()

        # Create an array of all plotted point coordinates [x, y]
        path_points = []
        for point in self.points:
            path_points.append([point.x, point.y])

        # Use matplotlib to create a path from the set of points.
        path = PlotPath(path_points)

        # Create a grid the same size as the image
        x, y = np.mgrid[:img.size[0], :img.size[1]]
        # Turn the grid into an array of points
        points = np.vstack((x.ravel(), y.ravel())).T

        # Create a boolean array of true/false values
        # True values are within the path, false outside the path.
        mask = path.contains_points(points)
        #path_points = points[np.where(mask)]

        # Create an image of the filled mask
        img_mask = mask.reshape(x.shape).T
        mask_image = Image.fromarray(img_mask)
        mask_image = mask_image.convert("RGB")
        mask_image.save("mask.png")

        # Apply a blur filter to the mask image
        blur = mask_image.copy()
        blur = blur.filter(ImageFilter.GaussianBlur(blurAmount))
        blur = blur.convert("L")
        blur.save("blurMask.png")

        # Apply the blurred mask to the image
        new_image = img.copy().convert("RGBA")
        new_image.putalpha(blur)

        # Create an opaque blank image the same size as the original image
        img = img.convert("RGBA")
        masked_image = Image.new("RGBA", img.size, color=(0, 0, 0, 0))
        # Paste the masked image into the blank image
        # Use the original image as another mask to retain transparent sections
        masked_image.paste(new_image, mask=img)
        masked_image.save("masked.png")

        # Save the cutout image
        new_image = qtg.QPixmap("masked.png")
        new_image.save(str(project_path / cropped_image_name))
        self.cw.active_image_item.setPixmap(new_image)

        post_cut_image = Image.open(str(project_path / cropped_image_name))
        post_cut_image.load()
        ActionManager.layerCut(self.cw.image_layer,
                               pre_cut_image, post_cut_image)

        self.cw.image_layer.applyAlterations()

    def smoothEdges(self):
        self.all_points = self.points

        # Create a list called point_pairs which takes the plotted points
        # points[0], points[1], points[2], points[3] ... points[n]
        # and creates a set of overlapping pairs:
        # [ [0, 1, 2], [1, 2, 3], [2, 3, 4] ... [n, n+1, n+2] ]
        self.zipped_points = zip(
            self.all_points, self.all_points[1:], self.all_points[2:])
        self.point_pairs = []

        for zipped_pair in self.zipped_points:
            pair = []
            for image_point in zipped_pair:
                pair.append(image_point)
            self.point_pairs.append(pair)

        # Remove all lines and points
        self.line_manager.removeAllLines()
        self.removeAllPoints()

        self.smooth_points = []
        self.counter = 0
        t_range = np.linspace(0, 1, 10)

        # For each set of 3 points calculate the coordinates of a
        # bezier curve between the points using t_range as the intervals
        # at which to find the coordinates.
        for zipped_point in self.point_pairs:
            self.counter += 1

            if (self.counter % 2 == 0):
                # Skip every 2nd set of points
                continue

            for t in t_range:
                # Calculate bezier coordinates
                px = (((1-t) ** 2) * zipped_point[0].x) + (2 * (1-t) *
                                                           t * zipped_point[1].x) + ((t ** 2) * zipped_point[2].x)
                py = (((1-t) ** 2) * zipped_point[0].y) + (2 * (1-t) *
                                                           t * zipped_point[1].y) + ((t ** 2) * zipped_point[2].y)
                self.smooth_points.append((round(px), round(py)))

        # Add the new set of points
        for smooth_point in self.smooth_points:
            self.addPoint(smooth_point[0], smooth_point[1])

        # Join the last point to the first point
        self.joinMask()


class CutoutGraphicsView(qtw.QGraphicsView):
    def __init__(self, cw):
        super().__init__()
        self.cw = cw

    def updateView(self):
        scene = self.scene()
        r = scene.sceneRect()
        self.fitInView(0, 0, r.width(), r.height(), qtc.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        self.updateView()

    def showEvent(self, event):
        if not event.spontaneous():
            self.updateView()

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        # Update the visible scene in the zoomed in view
        self.cw.mini_view.setView(scene_pos.x(), scene_pos.y(), self.cw)
        # Update the cursor indicator position in the zoomed in view
        self.cw.indicator_pixmap_item.setPos(scene_pos.x()*2, scene_pos.y()*2)


class MiniGraphicsView(qtw.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.fitInView(200-50, 200-50, 100, 100)

    def setView(self, x, y, cw):
        if (x > cw.imageW - 25):
            x = cw.imageW - 25
        if (y > cw.imageH - 25):
            y = cw.imageH - 25

        x = x*2
        y = y*2

        self.fitInView(x-50, y-50, 100, 100)


class MiniGraphicsScene(qtw.QGraphicsScene):
    def __init__(self):
        super().__init__()


class CutoutGraphicsScene(qtw.QGraphicsScene):
    def __init__(self, cw):
        super().__init__()
        self.currently_moving = None
        self.cw = cw
        self.point_manager = self.cw.getPointManager()

    def setView(self, view):
        self.view = view

    def getView(self):
        return self.view

    def mousePressEvent(self, event):
        # Add a path point at the coordinates clicked by the user
        self.point_manager.addPoint(event.scenePos().x(), event.scenePos().y())

    def mouseDoubleClickEvent(self, event):
        pass


class CutoutGraphicsItem(qtw.QGraphicsPixmapItem):
    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setAcceptHoverEvents(True)


class MainGraphicsView(qtw.QGraphicsView):
    def __init__(self):
        super().__init__()

    def updateView(self):
        scene = self.scene()
        r = scene.sceneRect()
        self.fitInView(0, 0, Canvas.width(), Canvas.height(),
                       qtc.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        self.updateView()

    def showEvent(self, event):
        if not event.spontaneous():
            self.updateView()


class MainGraphicsScene(qtw.QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.currently_moving = None
        self.crop_mode = False
        self.crop_box = None
        self.previous_crop_box = None
        self.drawing_mode = False
        self.orig_crop_point = None
        self.pen = qtg.QPen(qtg.QBrush(qtg.QColor(0, 0, 0, 255)), 2)

    def setView(self, view):
        self.view = view

    def getView(self):
        return self.view

    def setCropMode(self, cropmode):
        self.crop_mode = cropmode

    def getCurrentCropBox(self):
        if (self.previous_crop_box):
            return self.previous_crop_box

    def mousePressEvent(self, event):
        if (self.crop_mode is True):
            # The user has the crop tool selected.
            # Clicking on the canvas deletes the crop box that is
            # already drawn.
            self.deleteCropBox()
            # Store the coordinates the user began drawing the cropbox at
            self.orig_crop_point = event.scenePos()
            self.drawing_mode = True
            return

        # If the user is not in crop mode then the click event is
        # passed to the item (if there is an item) at those coordinates.
        item = self.itemAt(event.scenePos(), qtg.QTransform())
        if (item):
            self.currently_moving = item
            self.sendEvent(item, event)

    def deleteCropBox(self):
        if (self.previous_crop_box):
            self.removeItem(self.previous_crop_box)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if (self.crop_mode is True and self.drawing_mode is True):
            # The user has the crop tool selected and is currently drawing
            # a crop box. The width and height of the crop box is calculated
            # by the difference in the current mouse position and the original
            # coordinates the user clicked.
            crop_box_width = event.scenePos().x() - self.orig_crop_point.x()
            crop_box_height = event.scenePos().y() - self.orig_crop_point.y()

            if (self.crop_box is None):
                # The crop box has not yet been drawn.
                # Add a rectangle to the canvas visualising the crop box
                # for the user to see.
                self.crop_box = qtw.QGraphicsRectItem(
                    self.orig_crop_point.x(), self.orig_crop_point.y(), crop_box_width, crop_box_height)
                self.crop_box.setPen(self.pen)
                self.crop_box.setZValue(100)
                self.addItem(self.crop_box)
            # Update the crop box with the current width and height.
            self.crop_box.setRect(self.orig_crop_point.x(
            ), self.orig_crop_point.y(), crop_box_width, crop_box_height)
            return

        if (self.currently_moving):
            # If the user is currently moving a layer then pass the event
            # to the item.
            self.sendEvent(self.currently_moving, event)

    def mouseReleaseEvent(self, event):
        if (self.crop_mode is True):
            # The user is in crop mode. The crop box is
            # stored in previous_crop_box and drawing mode
            # is turned off.
            self.previous_crop_box = self.crop_box
            self.crop_box = None
            self.drawing_mode = False

        if (self.currently_moving):
            # The user has finished moving an item. The event is passed
            # down to the item.
            self.sendEvent(self.currently_moving, event)
            # Moving mode is turned off.
            self.currently_moving = None

        self.view.updateView()

    def mouseDoubleClickEvent(self, event):
        # Ignore double click events.
        pass


class CanvasRotateItem(qtw.QGraphicsPixmapItem):
    def __init__(self, layer):
        pixmap = qtg.QPixmap(":/icon_rotate.png")
        super().__init__(pixmap)
        self.layer = layer
        self.setAcceptHoverEvents(True)
        self.setScale(0.5)
        self.setZValue(100)

    def hoverEnterEvent(self, event):
        qtw.QApplication.setOverrideCursor(qtc.Qt.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        qtw.QApplication.setOverrideCursor(qtc.Qt.ArrowCursor)

    def mousePressEvent(self, event):
        # The user has grabbed the rotate icon and is beginning
        # the process to change the rotation.
        qtw.QApplication.setOverrideCursor(qtc.Qt.ClosedHandCursor)
        # The original angle before the user completes the action
        # is stored.
        self.orig_angle = self.layer.getRotation()

    def mouseMoveEvent(self, event):
        new_cursor_pos = event.scenePos()
        origin_pos = self.layer.mapToScene(self.layer.transformOriginPoint())

        # Calculate the difference along both the x and y axis
        # between the cursor position and the layer origin position.
        # Layer origin is the center of the layer image.
        diff_x = new_cursor_pos.x() - origin_pos.x()
        diff_y = new_cursor_pos.y() - origin_pos.y()

        # Calculate the angle using the x and y differences
        angle = ((math.atan2(diff_y, diff_x) / math.pi) * 180 + 45)

        # Round the angle to a mutliple of 15 if it is within
        # 1 degree of the multiple.
        angle_degree = round(angle) % 15
        if (angle_degree == 0 or angle_degree == 1 or angle_degree == 14):
            angle = round(angle)
            if angle_degree == 1:
                angle -= 1
            elif angle_degree == 14:
                angle += 1

        self.layer.setRotation(angle)

    def mouseReleaseEvent(self, event):
        qtw.QApplication.restoreOverrideCursor()
        # An action is logged containing the angle before and after
        # the user changed it.
        ActionManager.layerRotated(
            self.layer.image_layer, self.orig_angle, self.layer.getRotation())
        # Reposition the rotate icon
        self.positionIcon()

    def positionIcon(self):
        # Position the icon to be at the top right of the image layer
        posX = self.layer.sceneBoundingRect().topRight().x()
        posY = self.layer.sceneBoundingRect().topRight().y()

        # If any part of the icon sits outside of the canvas
        # it is repositioned to be fully visible.
        if (posX + self.sceneBoundingRect().width() > Canvas.width()):
            posX = Canvas.width() - self.sceneBoundingRect().width()
        elif (posX < self.sceneBoundingRect().width()):
            posX = self.sceneBoundingRect().width()
        if (posY + self.sceneBoundingRect().height() > Canvas.height()):
            posY = Canvas.height() - self.sceneBoundingRect().height()
        elif (posY < self.boundingRect().height()):
            posY = self.sceneBoundingRect().height()
        self.setPos(posX, posY)


class CanvasScaleItem(qtw.QGraphicsPixmapItem):
    def __init__(self, layer):
        pixmap = qtg.QPixmap(":/icon_resize.png")
        super().__init__(pixmap)
        self.layer = layer
        self.setAcceptHoverEvents(True)
        self.current_bounding_rect = self.layer.sceneBoundingRect()
        self.setScale(0.5)
        self.setZValue(100)

    def hoverEnterEvent(self, event):
        qtw.QApplication.setOverrideCursor(qtc.Qt.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        qtw.QApplication.setOverrideCursor(qtc.Qt.ArrowCursor)

    def mousePressEvent(self, event):
        # Store the original scale of the item before the user
        # completes the change action.
        self.orig_scale = self.layer.getScale()
        qtw.QApplication.setOverrideCursor(qtc.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        new_cursor_pos = event.scenePos()
        origin_pos = self.layer.mapToScene(self.layer.transformOriginPoint())

        # Calculate the x and y difference between the cursor position
        # and the origin of the layer image.
        diff_x = new_cursor_pos.x() - origin_pos.x()
        diff_y = new_cursor_pos.y() - origin_pos.y()

        # Calculate the difference between the layer's top right
        # coordinates and the origin coordinates.
        orig_diff_x = self.layer.boundingRect().topRight().x() - \
            self.layer.transformOriginPoint().x()
        orig_diff_y = self.layer.boundingRect().topRight().y() - \
            self.layer.transformOriginPoint().y()

        # Calculate the distance between the mouse and the origin
        mouse_distance = math.sqrt((diff_x ** 2) + (diff_y ** 2))
        # Calculate the distance between the top right and the origin
        original_distance = math.sqrt((orig_diff_x ** 2) + (orig_diff_y ** 2))

        # Calculate the scale as a ratio between the distances
        scale = mouse_distance / original_distance
        self.layer.setScale(scale)

    def mouseReleaseEvent(self, event):
        self.new_scale = self.layer.getScale()
        # Log an action containing the scale of the image before and
        # after the user changed it.
        ActionManager.layerScaled(
            self.layer.image_layer, self.orig_scale, self.new_scale)
        self.current_bounding_rect = self.layer.sceneBoundingRect()
        qtw.QApplication.restoreOverrideCursor()
        # Reposition the scale icon.
        self.positionIcon()

    def positionIcon(self):
        # Position the icon to be at the top right of the image layer
        posX = self.layer.sceneBoundingRect().topRight().x()
        posY = self.layer.sceneBoundingRect().topRight().y()

        # If any part of the icon sits outside of the canvas
        # it is repositioned to be fully visible.
        if (posX + self.sceneBoundingRect().width() > Canvas.width()):
            posX = Canvas.width() - self.sceneBoundingRect().width()
        elif (posX < self.sceneBoundingRect().width()):
            posX = self.sceneBoundingRect().width()
        if (posY + self.sceneBoundingRect().height() > Canvas.height()):
            posY = Canvas.height() - self.sceneBoundingRect().height()
        elif (posY < self.boundingRect().height()):
            posY = self.sceneBoundingRect().height()
        self.setPos(posX, posY)


class CanvasGraphicsItem(qtw.QGraphicsPixmapItem):
    def __init__(self, pixmap, layer):
        self.pixmap = pixmap
        self.image_layer = layer
        super().__init__(pixmap)
        self.setAcceptHoverEvents(True)
        self.setFlag(qtw.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.item_rotation = 0
        self.item_scale = 1

    def getRotation(self):
        return self.item_rotation

    def getScale(self):
        return self.item_scale

    def hoverEnterEvent(self, event):
        qtw.QApplication.setOverrideCursor(qtc.Qt.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        qtw.QApplication.setOverrideCursor(qtc.Qt.ArrowCursor)

    def mousePressEvent(self, event):
        # The user begins the process to move a layer.
        # Store the original x,y coordinates of the later.
        self.orig_pos_x = self.pos().x()
        self.orig_pos_y = self.pos().y()
        qtw.QApplication.setOverrideCursor(qtc.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        orig_cursor_pos = event.lastScenePos()
        new_cursor_pos = event.scenePos()

        # Calculate the difference between the current cursor position
        # and the cursors last known position.
        dif_x = new_cursor_pos.x() - orig_cursor_pos.x()
        dif_y = new_cursor_pos.y() - orig_cursor_pos.y()

        # Calculate the layer's new position by adding the
        # difference in cursor positions to the layer's current position.
        orig_pos = self.pos()
        new_x = dif_x + orig_pos.x()
        new_y = dif_y + orig_pos.y()

        self.setPos(qtc.QPointF(new_x, new_y))

    def mouseReleaseEvent(self, event):
        # The user has stopped moving the item.
        # Log an action containing the start and end position of the layer.
        ActionManager.layerMoved(
            self, self.orig_pos_x, self.orig_pos_y, self.pos().x(), self.pos().y())
        qtw.QApplication.restoreOverrideCursor()

    def setRotation(self, angle):
        super().setRotation(angle)
        self.item_rotation = angle

    def updateRotation(self):
        super().setRotation(self.item_rotation)

    def setScale(self, scale):
        super().setScale(scale)
        self.item_scale = scale

    def centreItem(self):
        # Position the item centrally in the canvas
        new_x_pos = (Canvas.width() - self.boundingRect().width()) / 2
        new_y_pos = (Canvas.height() - self.boundingRect().height()) / 2

        self.setPos(qtc.QPointF(new_x_pos, new_y_pos))

    def itemChange(self, change, value):
        return super().itemChange(change, value)


def setOriginToCenter(item):
    item.setTransformOriginPoint(item.boundingRect().center())


def setOriginToZero(item):
    item.setTransformOriginPoint(0, 0)


class LayerManager():
    layers_container = []
    active_layer = None
    num_layers = 0

    def createNewLayer(image, layer_name, layer_z, layer_x, layer_y):
        # Create a new ImageLayer and add it to the array of layers
        new_layer = ImageLayer(image, layer_name, layer_z, layer_x, layer_y)
        LayerManager.layers_container.append(new_layer)
        # increase the layer counter
        LayerManager.num_layers += 1
        return new_layer

    def disableAll():
        # Disable all layers in the canvas
        for layer in LayerManager.layers_container:
            layer.disableAll()

    def setActiveLayer(ImageLayer):
        # Store the new active layer and alert the main window
        # that the active layer has changed.
        LayerManager.active_layer = ImageLayer
        mw.activeChanged()

    def getActiveLayer():
        return LayerManager.active_layer

    def tempDeleteLayer(image_layer):
        # Completely hides the layer from the canvas without
        # delete the object.
        image_layer.disableAll()
        image_layer.disableVisible()
        image_layer.getLayerWidget().setVisible(False)
        image_layer.getRandomiseWidget().setVisible(False)

    def undoDeleteLayer(image_layer):
        # Unhides the layer when the user decides to undo
        # the deletion of a layer.
        layer_z_position = image_layer.getZPosition()
        for layer in LayerManager.layers_container:
            if layer.getZPosition() >= layer_z_position:
                layer.setZPosition(layer.getZPosition() + 1)
        LayerManager.layers_container.append(image_layer)
        image_layer.enableVisible()
        image_layer.getLayerWidget().setVisible(True)
        image_layer.getRandomiseWidget().setVisible(True)

    def getLayerWidget(image_layer):
        return ImageLayer.getLayerWidget(image_layer)

    def moveLayerUp(image_layer):
        # Move a layer's Z position up within the canvas
        image_layer_z = image_layer.getZPosition()
        # Check if the selected layer is already at the top position
        if image_layer_z == LayerManager.num_layers - 1:
            mw.status_bar.showMessage("Layer is already at the top...", 5000)
            return
        # Move the selected layer up one position
        image_layer.setZPosition(image_layer_z + 1)

        # Move the layer that was originally above the selected
        # layer down by 1.
        for layer in LayerManager.layers_container:
            if layer == image_layer:
                continue
            if layer.getZPosition() - 1 == image_layer_z:
                layer.setZPosition(layer.getZPosition() - 1)

        mw.status_bar.showMessage("Layer moved up...", 3000)

    def moveLayerDown(image_layer):
        # Move a layer's Z position down within the canvas
        image_layer_z = image_layer.getZPosition()
        # Check if the selected layer is already at the bottom position
        if image_layer_z == 0:
            mw.status_bar.showMessage(
                "Layer is already at the bottom...", 5000)
            return
        # Move the selected layer down one position
        image_layer.setZPosition(image_layer_z - 1)

        # Move the layer that was originally below the selected layer
        # down by 1.
        for layer in LayerManager.layers_container:
            if layer == image_layer:
                continue
            if layer.getZPosition() + 1 == image_layer_z:
                layer.setZPosition(layer.getZPosition() + 1)

        mw.status_bar.showMessage("Layer moved down...", 3000)

    def getImageName(image_layer):
        image_layer.getImageName()

    def getLayerName(image_layer):
        image_layer.getLayerName()

    def getXPosition(image_layer):
        image_layer.getXPosition()

    def getYPosition(image_layer):
        image_layer.getYPosition()


class ImageLayer:

    def __init__(self, image, layer_name, layer_z, layer_x, layer_y):
        self.image_name = image
        self.cropped_image_name = "cropped_" + self.image_name
        self.altered_image_name = "altered_" + self.image_name

        # Ensure that the cropped and altered version of the image file exist
        self.cropped_image = Image.open(str(project_path / self.image_name))
        self.cropped_image.save(str(project_path / self.cropped_image_name))
        self.altered_image = Image.open(str(project_path / self.image_name))
        self.altered_image.save(str(project_path / self.altered_image_name))

        self.layer_name = layer_name

        # Set the layer position
        self.layer_z_position = layer_z
        self.layer_x_position = layer_x
        self.layer_y_position = layer_y

        # Generate widget's to be displayed in the layer manager and
        # randomisation tool.
        self.layer_widget = None
        self.randomise_widget = None
        self.createNewLayerWidget()
        self.createNewRandomiseWidget()

        # Initialise layer properties
        self.cropped = False
        self.rgb = [1, 1, 1]
        self.bw = False
        self.blur = False
        self.sharpness = 1
        self.brightness = 1
        self.contrast = 1
        self.filters = []

        # Create and position the layer item for the canvas
        self.image_pixmap = qtg.QPixmap(str(project_path / self.image_name))
        self.image_item = CanvasGraphicsItem(self.image_pixmap, self)
        self.image_item.setPos(self.layer_x_position, self.layer_y_position)
        self.image_item.setZValue(self.layer_z_position)
        self.image_item.setEnabled(False)

        # Create rotate and scale icon items for the layer
        self.rotate_item = CanvasRotateItem(self.image_item)
        self.scale_item = CanvasScaleItem(self.image_item)

        self.disableAll()

    def createNewLayerWidget(self):
        # create a new layer widget
        new_layer_widget = LayerWidget(self)
        self.layer_widget = new_layer_widget

    def createNewRandomiseWidget(self):
        new_randomise_widget = RandomiseWidget(self)
        self.randomise_widget = new_randomise_widget

    def randomiseLayer(self):
        ActionManager.layerRandomiseStart(self)
        # Call the randomisation function for each property
        # that is not locked.
        if (self.randomise_widget.rotate_locked == False):
            self.randomiseRotation()
        if (self.randomise_widget.scale_locked == False):
            self.randomiseScale()
        if (self.randomise_widget.position_locked == False):
            self.randomisePosition(True)
        if (self.randomise_widget.blur_locked == False):
            self.randomiseBlur()
        if (self.randomise_widget.contrast_locked == False):
            self.randomiseContrast()
        if (self.randomise_widget.brightness_locked == False):
            self.randomiseBrightness()
        if (self.randomise_widget.sharpness_locked == False):
            self.randomiseSharpness()
        if (self.randomise_widget.baw_locked == False):
            self.randomiseBaw()
        if (self.randomise_widget.rgb_locked == False):
            self.randomiseRGB()
        self.applyAlterations()
        ActionManager.layerRandomiseEnd(self)
        mw.status_bar.showMessage("Layer randomised...", 5000)

    def randomiseBrightness(self):
        # Generate a brightness alteration factor betwen 0-2
        orig_brightness = self.brightness
        random_brightness_factor = random.uniform(0, 2)
        ActionManager.brightnessChanged(
            self, orig_brightness, random_brightness_factor)
        self.setBrightness(random_brightness_factor)

    def randomiseContrast(self):
        # Generate a contrast alteration factor betwen 0-2
        orig_contrast = self.contrast
        random_contrast_factor = random.uniform(0, 2)
        ActionManager.contrastChanged(
            self, orig_contrast, random_contrast_factor)
        self.setContrast(random_contrast_factor)

    def randomiseSharpness(self):
        # Generate a sharpness alteration factor betwen 0-2
        orig_sharpness = self.sharpness
        random_sharpness_factor = random.uniform(0, 2)
        ActionManager.sharpnessChanged(
            self, orig_sharpness, random_sharpness_factor)
        self.setSharpness(random_sharpness_factor)

    def randomiseRGB(self):
        # Generate an alteration factor between 0-2
        # for the red, green and blue bands of the image.
        orig_rgb = self.rgb
        layer_rgb = [orig_rgb[0], orig_rgb[1], orig_rgb[2]]
        random_r_value = random.uniform(0, 2)
        random_g_value = random.uniform(0, 2)
        random_b_value = random.uniform(0, 2)
        new_rgb = [random_r_value, random_g_value, random_b_value]
        ActionManager.rgbChanged(self, layer_rgb, new_rgb)
        self.setRGB(random_r_value, random_g_value, random_b_value)

    def randomiseBaw(self):
        # Generate a true or false value for the black
        # and white filter over the image.
        orig_baw = self.bw
        new_baw = bool(random.getrandbits(1))
        ActionManager.bawChanged(self, orig_baw, new_baw)
        self.baw = new_baw

    def randomiseBlur(self):
        # Generate a true or false value for the blur
        # filter over the image.
        orig_blur = self.blur
        new_blur = bool(random.getrandbits(1))
        ActionManager.blurChanged(self, orig_blur, new_blur)
        self.blur = new_blur

    def randomiseRotation(self):
        # Generate an angle between 0-360
        orig_angle = self.image_item.getRotation()
        random_angle = random.randint(0, 360)
        self.image_item.setRotation(random_angle)
        ActionManager.layerRotated(self, orig_angle, random_angle)

    def randomiseScale(self):
        # Generate a random scale factor
        # The scale limit is the maximum scale the image can
        # be while staying within the canvas borders.
        orig_scale = self.image_item.getScale()

        height_factor = Canvas.height() / self.image_item.boundingRect().height()
        width_factor = Canvas.width() / self.image_item.boundingRect().width()
        scale_factor_limit = (min(height_factor, width_factor))

        random_scale_factor = random.uniform(0, scale_factor_limit)
        ActionManager.layerScaled(self, orig_scale, random_scale_factor)
        self.image_item.setScale(random_scale_factor)

    def randomisePosition(self, within_canvas):
        # Randomly generate an x and y value to position the image at
        # If within_canvas parameter is true the generated coordinates
        # will be bounded to ensure the image stays within the canvas borders
        # where possible.
        if within_canvas:
            width_difference = (self.image_item.sceneBoundingRect(
            ).width() - self.image_item.boundingRect().width())
            height_difference = (self.image_item.sceneBoundingRect(
            ).height() - self.image_item.boundingRect().height())
            min_x = int(width_difference / 2)
            min_y = int(height_difference / 2)
            max_x = int(Canvas.width(
            ) - (self.image_item.sceneBoundingRect().width()) + (width_difference / 2))
            max_y = int(Canvas.height(
            ) - self.image_item.boundingRect().height() - (height_difference / 2))
        else:
            min_x = 0
            min_y = 0
            max_x = int(Canvas.width())
            max_y = int(Canvas.height())

        if max_x < min_x:
            min_x, max_x = max_x, min_x
        if max_y < min_y:
            min_y, max_y = max_y, min_y

        random_x = random.randint(min_x, max_x)
        random_y = random.randint(min_y, max_y)
        ActionManager.layerMoved(
            self.image_item, self.getXPosition(), self.getYPosition(), random_x, random_y)
        self.setXY(random_x, random_y)

    def applyAlterations(self):
        self.new_image = Image.open(
            str(project_path / self.cropped_image_name))
        self.new_image.save(str(project_path / ("altered_" + self.image_name)))
        self.new_name = "altered_" + self.image_name
        if self.rgb != [1, 1, 1]:
            # alter rgb
            new_image = alterRGB(
                self.new_name, self.rgb[0], self.rgb[1], self.rgb[2])
            new_image.save(str(project_path / self.new_name))
        if self.bw == True:
            # apply bw
            new_image = makeLayerBaW(self.new_name)
            new_image.save(str(project_path / self.new_name))
        if self.blur == True:
            new_image = blurImage(self.new_name)
            new_image.save(str(project_path / self.new_name))
        if self.sharpness != 1:
            # apply sharpness
            new_image = enhanceSharpness(self.new_name, self.sharpness)
            new_image.save(str(project_path / self.new_name))
        if self.brightness != 1:
            # apply brightness
            new_image = enhanceBrightness(self.new_name, self.brightness)
            new_image.save(str(project_path / self.new_name))
        if self.contrast != 1:
            # apply contrast
            new_image = enhanceContrast(self.new_name, self.contrast)
            new_image.save(str(project_path / self.new_name))

        self.altered_image_name = self.new_name
        self.updatePixmap()

    def updatePixmap(self):
        # Update the layer's image in the canvas and the image
        # displayed in the layer's associated widget thumnbails.
        self.image_pixmap = qtg.QPixmap(
            str(project_path / self.altered_image_name))
        self.image_item.setPixmap(self.image_pixmap)
        self.layer_widget.updateThumbnail()
        self.randomise_widget.updateThumbnail()

    def getLayerItem(self):
        return self.image_item

    def getImagePixmap(self):
        return self.image_pixmap

    def getDisplayImage(self):
        return self.altered_image_name

    def getSharpness(self):
        return self.sharpness

    def setSharpness(self, sharpness):
        self.sharpness = sharpness

    def getBrightness(self):
        return self.brightness

    def setBrightness(self, brightness):
        self.brightness = brightness

    def getContrast(self):
        return self.contrast

    def setContrast(self, contrast):
        self.contrast = contrast

    def getBW(self):
        return self.bw

    def setBW(self, bw):
        self.bw = bw

    def getBlur(self):
        return self.blur

    def setBlur(self, blur):
        self.blur = blur

    def setRGB(self, r, g, b):
        self.setR(r)
        self.setG(g)
        self.setB(b)

    def getR(self):
        return self.rgb[0]

    def setR(self, r):
        self.rgb[0] = r

    def getG(self):
        return self.rgb[1]

    def setG(self, g):
        self.rgb[1] = g

    def getB(self):
        return self.rgb[2]

    def setB(self, b):
        self.rgb[2] = b

    def getOriginalItem(self):
        return self.image_name

    def getRotateItem(self):
        return self.rotate_item

    def getScaleItem(self):
        return self.scale_item

    def getRandomiseWidget(self):
        return self.randomise_widget

    def getLayerWidget(self):
        return self.layer_widget

    def getImageName(self):
        return self.image_name

    def setImageName(self, image_name):
        self.image_name = image_name

    def getLayerName(self):
        return self.layer_name

    def getZPosition(self):
        return self.layer_z_position

    def setZPosition(self, z):
        self.layer_z_position = z
        self.image_item.setZValue(z)

    def getXPosition(self):
        return self.image_item.pos().x()

    def setXPosition(self, x):
        self.image_item.setX(x)

    def getYPosition(self):
        return self.image_item.pos().y()

    def setYPosition(self, y):
        self.image_item.setY(y)

    def setXY(self, x, y):
        self.setXPosition(x)
        self.setYPosition(y)

    def disableAll(self):
        # Disbale rotating, scaling and moving the image.
        self.disableRotate()
        self.disableScale()
        self.disableDrag()

    def enableRotate(self):
        # Enable rotating the image
        self.rotate_item.positionIcon()
        self.rotate_item.setEnabled(True)
        self.rotate_item.setVisible(True)
        # Disable moving and scaling the image
        self.disableDrag()
        self.disableScale()

    def disableRotate(self):
        # Disbale rotating the image
        self.rotate_item.setEnabled(False)
        self.rotate_item.setVisible(False)

    def enableScale(self):
        # Enable scaling the image
        self.scale_item.positionIcon()
        self.scale_item.setEnabled(True)
        self.scale_item.setVisible(True)
        # Disable rotating and moving the image
        self.disableRotate()
        self.disableDrag()

    def disableScale(self):
        # Disable scaling the image
        self.scale_item.setEnabled(False)
        self.scale_item.setVisible(False)

    def enableDrag(self):
        # Enable moving the image
        self.image_item.setEnabled(True)
        # Disable rotating and scaling the image
        self.disableRotate()
        self.disableScale()

    def disableDrag(self):
        # Disbale moving the image
        self.image_item.setEnabled(False)

    def enableVisible(self):
        self.image_item.setVisible(True)

    def disableVisible(self):
        self.image_item.setVisible(False)

    def crop(self, x1, y1, x2, y2, originPoint):
        # Create an image containing the rotated, scaled and altered
        # version of the original image.
        crop_view = qtw.QGraphicsView()
        crop_view.setStyleSheet("background: transparent")
        crop_scene = qtw.QGraphicsScene()
        crop_view.setScene(crop_scene)

        crop_pixmap = qtg.QPixmap(str(project_path / self.cropped_image_name))
        crop_item = crop_scene.addPixmap(crop_pixmap)
        crop_item.setRotation(self.getLayerItem().getRotation())
        crop_item.setScale(self.getLayerItem().getScale())
        crop_scene.setSceneRect(qtc.QRectF())
        # Create a blank image with transparent background
        temp_image = qtg.QImage(crop_scene.sceneRect(
        ).size().toSize(), qtg.QImage.Format_ARGB32)
        temp_image.fill(0)

        # Paint the altered image into the blank image.
        painter2 = qtg.QPainter(temp_image)
        crop_scene.render(painter2)
        painter2.end()
        temp_image.save(str(project_path / ("pre_crop_" + self.image_name)))

        image = Image.open(str(project_path / ("pre_crop_" + self.image_name)))
        orig_image = image
        orig_image.load()

        cropped_image = cropImage(image, x1, y1, x2, y2)
        new_image = cropped_image
        new_image.load()
        cropped_image.save(str(project_path / ("cropped_" + self.image_name)))

        self.getLayerItem().setRotation(0)
        self.getLayerItem().setScale(1)

        orig_x, orig_y = self.getXPosition(), self.getYPosition()
        new_x, new_y = originPoint.x(), originPoint.y()
        coordinates = [orig_x, orig_y, new_x, new_y]
        ActionManager.layerCropped(self, orig_image, new_image, coordinates)

        self.setXY(new_x, new_y)
        self.applyAlterations()
        setOriginToCenter(self.getLayerItem())

    def alignVTop(self):
        orig_x, orig_y = self.getXPosition(), self.getYPosition()
        height_difference = (self.image_item.sceneBoundingRect(
        ).height() - self.image_item.boundingRect().height())
        x = self.getXPosition()
        y = 0 + (height_difference / 2)
        ActionManager.layerMoved(self.image_item, orig_x, orig_y, x, y)
        self.image_item.setPos(x, y)

    def alignVCenter(self):
        orig_x, orig_y = self.getXPosition(), self.getYPosition()
        x = self.getXPosition()
        y = (Canvas.height() - self.image_item.boundingRect().height()) / 2
        ActionManager.layerMoved(self.image_item, orig_x, orig_y, x, y)
        self.image_item.setPos(x, y)

    def alignVBottom(self):
        orig_x, orig_y = self.getXPosition(), self.getYPosition()
        height_difference = (self.image_item.sceneBoundingRect(
        ).height() - self.image_item.boundingRect().height())
        x = self.getXPosition()
        y = Canvas.height() - self.image_item.boundingRect().height() - \
            (height_difference / 2)
        ActionManager.layerMoved(self.image_item, orig_x, orig_y, x, y)
        self.image_item.setPos(x, y)

    def alignHLeft(self):
        orig_x, orig_y = self.getXPosition(), self.getYPosition()
        width_difference = (self.image_item.sceneBoundingRect().width() -
                            self.image_item.boundingRect().width())
        x = 0 + (width_difference / 2)
        y = self.getYPosition()
        ActionManager.layerMoved(self.image_item, orig_x, orig_y, x, y)
        self.image_item.setPos(x, y)

    def alignHCenter(self):
        orig_x, orig_y = self.getXPosition(), self.getYPosition()
        x = (Canvas.width() - self.image_item.boundingRect().width()) / 2
        y = self.getYPosition()
        ActionManager.layerMoved(self.image_item, orig_x, orig_y, x, y)
        self.image_item.setPos(x, y)

    def alignHRight(self):
        orig_x, orig_y = self.getXPosition(), self.getYPosition()
        width_difference = (self.image_item.sceneBoundingRect().width() -
                            self.image_item.boundingRect().width())
        x = Canvas.width() - (self.image_item.sceneBoundingRect().width()) + \
            (width_difference / 2)
        y = self.getYPosition()
        ActionManager.layerMoved(self.image_item, orig_x, orig_y, x, y)
        self.image_item.setPos(x, y)


class RandomiseWidget(qtw.QWidget):
    # A widget displaying the layer information
    # and randomisation options.

    title_font = qtg.QFont()
    title_font.setBold(True)

    def __init__(self, imgLayer, *args, **kwargs):
        super().__init__()
        self.image_layer = imgLayer
        self.is_layer_locked = True
        self.thumbnail_name = str(project_path / imgLayer.getImageName())
        self.layer_thumbnail_size = 50
        self.layer_name = imgLayer.getLayerName()
        self.position_locked = True
        self.rgb_locked = True
        self.blur_locked = True
        self.contrast_locked = True
        self.brightness_locked = True
        self.sharpness_locked = True
        self.baw_locked = True
        self.scale_locked = True
        self.rotate_locked = True
        self.options_visible = False

        # Container layout for the widget
        self.widget_layout = qtw.QVBoxLayout()
        self.widget_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.widget_layout)

        # Top section layout for the randomisation widget
        self.layer_top_section = qtw.QWidget()
        self.layer_layout = qtw.QHBoxLayout()
        self.layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_top_section.setLayout(self.layer_layout)

        # Layer details widget
        self.layer_details = qtw.QWidget()
        self.layer_details_layout = qtw.QVBoxLayout()
        self.layer_details.setSizePolicy(
            qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Expanding)
        self.layer_details.setLayout(self.layer_details_layout)

        # Layer options widget
        self.layer_options = qtw.QWidget()
        self.layer_options_layout = qtw.QHBoxLayout()
        self.layer_options_layout.setAlignment(qtc.Qt.AlignCenter)
        self.layer_options.setLayout(self.layer_options_layout)

        # Layer name label
        self.layer_name = qtw.QLabel(self.layer_name)
        self.layer_name.setAlignment(qtc.Qt.AlignCenter)
        self.layer_name.setFont(LayerWidget.title_font)

        # Layer thumbnail
        self.layer_img = qtg.QPixmap(self.thumbnail_name).scaled(
            self.layer_thumbnail_size, self.layer_thumbnail_size, qtc.Qt.KeepAspectRatio)
        self.layer_thumbnail_container_size = 75
        self.thumbnail_label = qtw.QLabel()
        self.thumbnail_label.setFixedSize(
            self.layer_thumbnail_container_size, self.layer_thumbnail_container_size)
        self.thumbnail_label.setStyleSheet("background-color: white;")
        self.thumbnail_label.setAlignment(qtc.Qt.AlignCenter)
        self.thumbnail_label.setPixmap(self.layer_img)

        # Option icon size
        layer_options_icon_size = 32
        # Set the layer locked and unlocked icons
        self.layer_locked = qtg.QPixmap(":/icon_locked.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        self.layer_unlocked = qtg.QPixmap(":/icon_unlocked.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        # Set the layer options hidden/unhidden icons
        self.options_shown = qtg.QPixmap(":/icon_visible.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        self.options_hidden = qtg.QPixmap(":/icon_invisible.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        # Set the randomise icon
        self.randomise_icon = qtg.QPixmap(":/icon_random.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)

        # Lock layer option
        self.layer_lock_label = ClickableLabel()
        self.layer_lock_label.setPixmap(self.layer_locked)
        self.layer_lock_label.clicked.connect(self.toggleLayerLock)

        # Hide/show layer options option
        self.options_visible_label = ClickableLabel()
        self.options_visible_label.setPixmap(self.options_hidden)
        self.options_visible_label.clicked.connect(self.toggleOptionsVisible)

        # Randomise layer option
        self.randomise_label = ClickableLabel()
        self.randomise_label.setPixmap(self.randomise_icon)
        self.randomise_label.clicked.connect(self.randomiseLayer)

        # Add the components to the randomise widget
        self.layer_details_layout.addWidget(self.layer_name)
        self.layer_details_layout.addWidget(self.layer_options)
        self.layer_options_layout.addWidget(self.layer_lock_label)
        self.layer_options_layout.addWidget(self.options_visible_label)
        self.layer_options_layout.addWidget(self.randomise_label)

        # Add the thumbnail and layer information to the widget
        self.layer_layout.addWidget(self.thumbnail_label)
        self.layer_layout.addWidget(self.layer_details)

        # Create the randomise_options_widget to contain the
        # options for each layer property
        self.randomise_options_widget = qtw.QWidget()
        self.randomise_options_layout = qtw.QGridLayout()
        self.randomise_options_widget.setLayout(self.randomise_options_layout)

        layer_options_icon_size_small = 25
        self.option_locked = qtg.QPixmap(":/icon_locked.png").scaled(
            layer_options_icon_size_small, layer_options_icon_size_small, qtc.Qt.KeepAspectRatio)
        self.option_unlocked = qtg.QPixmap(":/icon_unlocked.png").scaled(
            layer_options_icon_size_small, layer_options_icon_size_small, qtc.Qt.KeepAspectRatio)

        self.position_label = qtw.QLabel("Position")
        self.position_lock_label = ClickableLabel()
        self.position_lock_label.setPixmap(self.option_locked)
        self.position_lock_label.clicked.connect(self.togglePositionLock)

        self.scale_label = qtw.QLabel("Size")
        self.scale_lock_label = ClickableLabel()
        self.scale_lock_label.setPixmap(self.option_locked)
        self.scale_lock_label.clicked.connect(self.toggleScaleLock)

        self.rotate_label = qtw.QLabel("Rotation")
        self.rotate_lock_label = ClickableLabel()
        self.rotate_lock_label.setPixmap(self.option_locked)
        self.rotate_lock_label.clicked.connect(self.toggleRotateLock)

        self.blur_label = qtw.QLabel("Blur")
        self.blur_lock_label = ClickableLabel()
        self.blur_lock_label.setPixmap(self.option_locked)
        self.blur_lock_label.clicked.connect(self.toggleBlurLock)

        self.rgb_label = qtw.QLabel("Colours")
        self.rgb_lock_label = ClickableLabel()
        self.rgb_lock_label.setPixmap(self.option_locked)
        self.rgb_lock_label.clicked.connect(self.toggleRGBLock)

        self.brightness_label = qtw.QLabel("Brightness")
        self.brightness_lock_label = ClickableLabel()
        self.brightness_lock_label.setPixmap(self.option_locked)
        self.brightness_lock_label.clicked.connect(self.toggleBrightnessLock)

        self.contrast_label = qtw.QLabel("Contrast")
        self.contrast_lock_label = ClickableLabel()
        self.contrast_lock_label.setPixmap(self.option_locked)
        self.contrast_lock_label.clicked.connect(self.toggleContrastLock)

        self.sharpness_label = qtw.QLabel("Sharpness")
        self.sharpness_lock_label = ClickableLabel()
        self.sharpness_lock_label.setPixmap(self.option_locked)
        self.sharpness_lock_label.clicked.connect(self.toggleSharpnessLock)

        self.baw_label = qtw.QLabel("Black & White")
        self.baw_lock_label = ClickableLabel()
        self.baw_lock_label.setPixmap(self.option_locked)
        self.baw_lock_label.clicked.connect(self.toggleBawLock)

        self.randomise_options_layout.addWidget(
            self.position_label, 0, 0, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.position_lock_label, 0, 1, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.scale_label, 1, 0, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.scale_lock_label, 1, 1, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.rotate_label, 2, 0, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.rotate_lock_label, 2, 1, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.blur_label, 3, 0, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.blur_lock_label, 3, 1, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.rgb_label, 4, 0, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.rgb_lock_label, 4, 1, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.brightness_label, 5, 0, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.brightness_lock_label, 5, 1, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.contrast_label, 6, 0, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.contrast_lock_label, 6, 1, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.sharpness_label, 7, 0, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.sharpness_lock_label, 7, 1, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.baw_label, 8, 0, alignment=qtc.Qt.AlignCenter)
        self.randomise_options_layout.addWidget(
            self.baw_lock_label, 8, 1, alignment=qtc.Qt.AlignCenter)

        # Set the style of the widget
        self.layer_top_section.setStyleSheet("background-color: white;")
        self.randomise_options_widget.setStyleSheet(
            "background-color: #e8f7fa;")

        self.widget_layout.addWidget(
            self.layer_top_section, alignment=qtc.Qt.AlignCenter)
        self.widget_layout.addWidget(self.randomise_options_widget)

        # Hide the randomisation options
        self.randomise_options_widget.setVisible(False)

    def randomiseLayer(self):
        # If the layer is not locked then the layer's randomisation
        # function is called
        if (self.is_layer_locked == False):
            self.image_layer.randomiseLayer()
        else:
            mw.status_bar.showMessage("Layer is locked...", 4000)

    def toggleLayerLock(self):
        if (self.is_layer_locked):
            self.is_layer_locked = False
            self.layer_lock_label.setPixmap(self.layer_unlocked)
            self.options_visible = True
            self.options_visible_label.setPixmap(self.options_shown)
            self.randomise_options_widget.setVisible(True)
            mw.status_bar.showMessage("Layer unlocked...", 2000)
        else:
            self.is_layer_locked = True
            self.layer_lock_label.setPixmap(self.layer_locked)
            self.options_visible = False
            self.options_visible_label.setPixmap(self.options_hidden)
            self.randomise_options_widget.setVisible(False)
            mw.status_bar.showMessage("Layer locked...", 2000)

    def toggleOptionsVisible(self):
        if (self.options_visible):
            self.options_visible = False
            self.randomise_options_widget.setVisible(False)
            self.options_visible_label.setPixmap(self.options_hidden)
            mw.status_bar.showMessage("Options hidden...", 2000)
        else:
            self.options_visible = True
            self.randomise_options_widget.setVisible(True)
            self.options_visible_label.setPixmap(self.options_shown)
            mw.status_bar.showMessage("Options shown...", 2000)

    def toggleRotateLock(self):
        if (self.rotate_locked):
            self.rotate_locked = False
            self.rotate_lock_label.setPixmap(self.option_unlocked)
            mw.status_bar.showMessage("Option unlocked...", 2000)
        else:
            self.rotate_locked = True
            self.rotate_lock_label.setPixmap(self.option_locked)
            mw.status_bar.showMessage("Option locked...", 2000)

    def toggleBlurLock(self):
        if (self.blur_locked):
            self.blur_locked = False
            self.blur_lock_label.setPixmap(self.option_unlocked)
            mw.status_bar.showMessage("Option unlocked...", 2000)
        else:
            self.blur_locked = True
            self.blur_lock_label.setPixmap(self.option_locked)
            mw.status_bar.showMessage("Option locked...", 2000)

    def toggleBrightnessLock(self):
        if (self.brightness_locked):
            self.brightness_locked = False
            self.brightness_lock_label.setPixmap(self.option_unlocked)
            mw.status_bar.showMessage("Option locked...", 2000)
        else:
            self.brightness_locked = True
            self.brightness_lock_label.setPixmap(self.option_locked)
            mw.status_bar.showMessage("Option locked...", 2000)

    def toggleContrastLock(self):
        if (self.contrast_locked):
            self.contrast_locked = False
            self.contrast_lock_label.setPixmap(self.option_unlocked)
            mw.status_bar.showMessage("Option unlocked...", 2000)
        else:
            self.contrast_locked = True
            self.contrast_lock_label.setPixmap(self.option_locked)
            mw.status_bar.showMessage("Option locked...", 2000)

    def toggleSharpnessLock(self):
        if (self.sharpness_locked):
            self.sharpness_locked = False
            self.sharpness_lock_label.setPixmap(self.option_unlocked)
            mw.status_bar.showMessage("Option unlocked...", 2000)
        else:
            self.sharpness_locked = True
            self.sharpness_lock_label.setPixmap(self.option_locked)
            mw.status_bar.showMessage("Option locked...", 2000)

    def toggleBawLock(self):
        if (self.baw_locked):
            self.baw_locked = False
            self.baw_lock_label.setPixmap(self.option_unlocked)
            mw.status_bar.showMessage("Option unlocked...", 2000)
        else:
            self.baw_locked = True
            self.baw_lock_label.setPixmap(self.option_locked)
            mw.status_bar.showMessage("Option locked...", 2000)

    def toggleRGBLock(self):
        if (self.rgb_locked):
            self.rgb_locked = False
            self.rgb_lock_label.setPixmap(self.option_unlocked)
            mw.status_bar.showMessage("Option unlocked...", 2000)
        else:
            self.rgb_locked = True
            self.rgb_lock_label.setPixmap(self.option_locked)
            mw.status_bar.showMessage("Option locked...", 2000)

    def togglePositionLock(self):
        if (self.position_locked):
            self.position_locked = False
            self.position_lock_label.setPixmap(self.option_unlocked)
            mw.status_bar.showMessage("Option unlocked...", 2000)
        else:
            self.position_locked = True
            self.position_lock_label.setPixmap(self.option_locked)
            mw.status_bar.showMessage("Option locked...", 2000)

    def toggleScaleLock(self):
        if (self.scale_locked):
            self.scale_locked = False
            self.scale_lock_label.setPixmap(self.option_unlocked)
            mw.status_bar.showMessage("Option unlocked...", 2000)
        else:
            self.scale_locked = True
            self.scale_lock_label.setPixmap(self.option_locked)
            mw.status_bar.showMessage("Option locked...", 2000)

    def updateThumbnail(self):
        self.layer_img = self.image_layer.getImagePixmap().scaled(
            self.layer_thumbnail_size, self.layer_thumbnail_size, qtc.Qt.KeepAspectRatio)
        self.thumbnail_label.setPixmap(self.layer_img)


class LayerWidget(qtw.QWidget):
    # Create a widget contain a layer's information and options
    title_font = qtg.QFont()
    title_font.setBold(True)

    def __init__(self, imgLayer, *args, **kwargs):
        super().__init__()
        # Set the values for the widget
        self.image_layer = imgLayer
        self.is_layer_active = False
        self.is_layer_visible = True
        self.thumbnail_name = str(project_path / imgLayer.getImageName())
        self.layer_thumbnail_size = 50
        self.layer_name = imgLayer.getLayerName()

        # Layer layout
        # Create a horizontal layout with 0 margins
        self.layer_layout = qtw.QHBoxLayout()
        self.layer_layout.setContentsMargins(0, 0, 0, 0)
        # Assign layout to the layer widget
        self.setLayout(self.layer_layout)

        # Layer details
        # Create a vertical layout
        self.layer_details = qtw.QWidget()
        self.layer_details_layout = qtw.QVBoxLayout()
        self.layer_details_layout.setContentsMargins(0, 0, 0, 0)
        # Assign vertical layout to layer details container
        self.layer_details.setLayout(self.layer_details_layout)

        # Layer options
        # Create a horizontal layout
        self.layer_options = qtw.QWidget()
        self.layer_options_layout = qtw.QHBoxLayout()
        # Center the layout options container
        self.layer_options_layout.setAlignment(qtc.Qt.AlignCenter)
        # Assign horizontal layout to the layer options container
        self.layer_options.setLayout(self.layer_options_layout)

        # Layer name and options widgets
        self.layer_name = qtw.QLabel(self.layer_name)
        self.layer_name.setAlignment(qtc.Qt.AlignCenter)
        self.layer_name.setFont(LayerWidget.title_font)

        # Thumbnail widget
        # Set thumbnail size
        layer_thumbnail_size = 50
        self.layer_img = qtg.QPixmap(self.thumbnail_name).scaled(
            layer_thumbnail_size, layer_thumbnail_size, qtc.Qt.KeepAspectRatio)
        # Set thumbnail container size
        layer_thumbnail_container_size = 75
        self.thumbnail_label = qtw.QLabel()
        self.thumbnail_label.setFixedSize(
            layer_thumbnail_container_size, layer_thumbnail_container_size)
        self.thumbnail_label.setStyleSheet("background-color: white;")
        self.thumbnail_label.setAlignment(qtc.Qt.AlignCenter)
        self.thumbnail_label.setPixmap(self.layer_img)

        # Set layer options icon size
        layer_options_icon_size = 32
        # Move layer down option
        self.layer_down = qtg.QPixmap(":/icon_down_arrow.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        self.layer_down_label = ClickableLabel()
        self.layer_down_label.setPixmap(self.layer_down)
        self.layer_down_label.clicked.connect(self.moveLayerDown)

        # Move layer up option
        self.layer_up = qtg.QPixmap(":/icon_up_arrow.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        self.layer_up_label = ClickableLabel()
        self.layer_up_label.setPixmap(self.layer_up)
        self.layer_up_label.clicked.connect(self.moveLayerUp)

        # Toggle active layer option
        self.layer_active_on = qtg.QPixmap(":/icon_switch_on.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        self.layer_active_off = qtg.QPixmap(":/icon_switch_off.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        self.layer_active_label = ClickableLabel()
        self.layer_active_label.setPixmap(self.layer_active_off)
        self.layer_active_label.clicked.connect(self.toggleLayerActive)

        # Toggle visible layer option
        self.layer_visible_on = qtg.QPixmap(":/icon_visible.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        self.layer_visible_off = qtg.QPixmap(":/icon_invisible.png").scaled(
            layer_options_icon_size, layer_options_icon_size, qtc.Qt.KeepAspectRatio)
        self.layer_visible_label = ClickableLabel()
        self.layer_visible_label.setPixmap(self.layer_visible_on)
        self.layer_visible_label.clicked.connect(self.toggleLayerVisible)

        # Populate widget
        self.layer_details_layout.addWidget(self.layer_name)
        self.layer_details_layout.addWidget(self.layer_options)
        self.layer_options_layout.addWidget(self.layer_down_label)
        self.layer_options_layout.addWidget(self.layer_active_label)
        self.layer_options_layout.addWidget(self.layer_visible_label)
        self.layer_options_layout.addWidget(self.layer_up_label)

        self.layer_layout.addWidget(self.thumbnail_label)
        self.layer_layout.addWidget(self.layer_details)

    def updateThumbnail(self):
        self.layer_img = self.image_layer.getImagePixmap().scaled(
            self.layer_thumbnail_size, self.layer_thumbnail_size, qtc.Qt.KeepAspectRatio)
        self.thumbnail_label.setPixmap(self.layer_img)

    def returnLayerName(self):
        return self.layer_name.text()

    def moveLayerDown(self):
        # Pass the image layer to LayerManager to be moved down
        ActionManager.layerMovedDown(self.image_layer)
        LayerManager.moveLayerDown(self.image_layer)

    def moveLayerUp(self):
        # Pass the image layer to LayerManager to be moved up
        ActionManager.layerMovedUp(self.image_layer)
        LayerManager.moveLayerUp(self.image_layer)

    def toggleLayerActive(self):
        orig_active = LayerManager.getActiveLayer()
        if (self.is_layer_active):
            # Turn the active layer off
            self.layerActiveOff()
            mw.status_bar.showMessage("Layer deactivated...", 3000)
        else:
            # Make this layer the active layer
            # Make current active layer inactive
            self.layerActiveOn()
            mw.status_bar.showMessage("Layer activated...", 3000)
        new_active = LayerManager.getActiveLayer()
        ActionManager.activeLayerChanged(orig_active, new_active)

    def layerActiveOn(self):
        if (LayerManager.getActiveLayer() is not None):
            # Turn the currently active layer off
            LayerManager.getActiveLayer().getLayerWidget().layerActiveOff()
        # Set the widget's layer as active
        LayerManager.setActiveLayer(self.image_layer)
        self.is_layer_active = True
        self.layer_active_label.setPixmap(self.layer_active_on)

    def layerActiveOff(self):
        # Deactivate the widget's layer
        LayerManager.setActiveLayer(None)
        self.is_layer_active = False
        self.layer_active_label.setPixmap(self.layer_active_off)

    def toggleLayerVisible(self):
        orig_visible = self.is_layer_visible
        if (self.is_layer_visible):
            self.turnVisibleOff()
            mw.status_bar.showMessage("Layer hidden...", 3000)
        else:
            self.turnVisibleOn()
            mw.status_bar.showMessage("Layer visible...", 3000)
        new_visible = self.is_layer_visible
        ActionManager.layerVisibleChanged(self, orig_visible, new_visible)

    def turnVisibleOff(self):
        self.image_layer.disableVisible()
        self.is_layer_visible = False
        self.layer_visible_label.setPixmap(self.layer_visible_off)

    def turnVisibleOn(self):
        self.image_layer.enableVisible()
        self.is_layer_visible = True
        self.layer_visible_label.setPixmap(self.layer_visible_on)


class ClickableLabel(qtw.QLabel):
    # A subclass of QLabel that emits a signal when the label is pressed
    clicked = qtc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args)

    def mousePressEvent(self, ev):
        self.clicked.emit()


class ColourLabel(ClickableLabel):
    colourClicked = qtc.pyqtSignal(str)

    def __init__(self, colour, index):
        super().__init__()
        self.colour = colour
        self.active = False
        self.index = index
        self.border = "0"

    def getLabelColour(self):
        return self.colour

    def setLabelColour(self, colour):
        self.colour = colour

    def mousePressEvent(self, ev):
        # When pressed emits a signal container the label's colour value
        self.colourClicked.emit(self.colour)
        if (self.active):
            # If the label is active and pressed then the
            # colour choice is cleared.
            if (self.index == 1):
                GradientManager.clearActiveLabel1()
            elif (self.index == 2):
                GradientManager.clearActiveLabel1()
        # Toggle the layer's active property
        self.toggleActive()

    def toggleActive(self):
        if (self.active):
            # Deactivate
            self.active = False
            self.border = "0"
            self.updateStyleSheet()
        else:
            # Activate
            self.active = True
            self.border = "1"
            self.updateStyleSheet()
            self.setActive()

    def setActive(self):
        if (self.index == 1):
            GradientManager.setActiveLabel1(self)
        elif (self.index == 2):
            GradientManager.setActiveLabel2(self)

    def updateStyleSheet(self):
        self.setStyleSheet("background-color: rgb" + self.colour +
                           "; border: " + self.border + "px solid black;")


class ToolTitleWidget(qtw.QWidget):
    # A widget to display a title and a help icon
    # The help icon shows a tooltip when hovered over
    option_icon_size = qtc.QSize(25, 25)

    def __init__(self, name):
        super().__init__()
        self.tool_name = name
        self.help_text = ""

        self.title_widget = qtw.QWidget()
        self.title_layout = qtw.QVBoxLayout()

        self.title_label = qtw.QLabel("<b>" + self.tool_name + "</b>")
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(qtc.Qt.AlignCenter)

        self.tooltip = qtw.QPushButton()
        self.tooltip.setToolTip(self.help_text)
        self.tooltip.setFlat(True)
        self.tooltip.setIcon(qtg.QIcon(':/icon_help.png'))
        self.tooltip.setIconSize(ToolTitleWidget.option_icon_size)

        self.title_widget.setLayout(self.title_layout)
        self.title_layout.addWidget(self.tooltip)
        self.title_layout.addWidget(self.title_label)

        self.setLayout(self.title_layout)

    def setHelpText(self, text):
        self.help_text = text
        self.tooltip.setToolTip(self.help_text)


class GradientManager():
    # A class to manage the active colours selected for the gradient tool
    active_label_1 = None
    active_label_2 = None
    active_1_widget = None
    active_2_widget = None

    def setActiveLabel1(label):
        if GradientManager.active_label_1:
            GradientManager.active_label_1.toggleActive()
        GradientManager.active_label_1 = label
        GradientManager.updateActive1Widget()

    def setActiveLabel2(label):
        if GradientManager.active_label_2:
            GradientManager.active_label_2.toggleActive()
        GradientManager.active_label_2 = label
        GradientManager.updateActive2Widget()

    def setActive1Widget(widget):
        GradientManager.active_1_widget = widget
        GradientManager.updateActive1Widget()

    def setActive2Widget(widget):
        GradientManager.active_2_widget = widget
        GradientManager.updateActive2Widget()

    def updateActive1Widget():
        if GradientManager.active_label_1:
            colour = GradientManager.active_label_1.getLabelColour()
        else:
            colour = "(255,255,255)"
        GradientManager.active_1_widget.setStyleSheet(
            "background-color: rgb" + colour + "; border: 1px solid black;")

    def updateActive2Widget():
        if GradientManager.active_label_2:
            colour = GradientManager.active_label_2.getLabelColour()
        else:
            colour = "(255,255,255)"
        GradientManager.active_2_widget.setStyleSheet(
            "background-color: rgb" + colour + "; border: 1px solid black;")

    def getActiveLabel1(self):
        return GradientManager.active_label_1

    def getActiveLabel2(self):
        return GradientManager.active_label_2

    def clearActiveLabel1():
        GradientManager.active_label_1 = None
        GradientManager.updateActive1Widget()

    def clearActiveLabel2():
        GradientManager.active_label_2 = None
        GradientManager.updateActive2Widget()

    def getActiveColour(index):
        if index == 1:
            if GradientManager.active_label_1:
                return GradientManager.active_label_1.getLabelColour()
            else:
                return "(255,255,255)"
        if index == 2:
            if GradientManager.active_label_2:
                return GradientManager.active_label_2.getLabelColour()
            else:
                return "(255,255,255)"


class Canvas():
    def __init__(self, w, h):
        Canvas.w = w
        Canvas.h = h

    def height():
        return Canvas.h

    def width():
        return Canvas.w


class CutoutWindow(qtw.QWidget):
    def __init__(self, imageCutoutLayer):
        super().__init__()

        self.setWindowTitle('Digital Collage Creator - Cutout Mode')
        self.setWindowIcon(qtg.QIcon(':/icon_logo.png'))
        self.setStyleSheet("background-color: #cdf1f9;")

        self.point_manager = PointManager(self)
        self.line_manager = LineManager(self)
        self.point_manager.setLineManager()

        self.image_layer = imageCutoutLayer
        self.cropped_image_name = self.image_layer.cropped_image_name
        self.image_path = str(project_path / self.cropped_image_name)
        self.imageW, self.imageH = Image.open(self.image_path).size

        self.setLayout(qtw.QHBoxLayout())

        self.canvas = (500, 500)
        # Create the view to display the layer image
        self.cutout_view = CutoutGraphicsView(self)
        self.cutout_view.setFixedSize(500, 500)
        self.cutout_view.setViewportMargins(-2, -2, -2, -2)
        self.layout().addWidget(self.cutout_view)
        self.cutout_view.setStyleSheet("background-color: white;")

        # Create the mini view to display the zoomed in view of the canvas
        self.mini_view = MiniGraphicsView()
        self.mini_view.setFixedSize(100, 100)
        self.mini_view.setViewportMargins(-2, -2, -2, -2)

        # Create a widget to contain the mini view and cutout options
        self.side_container = qtw.QWidget()
        self.side_layout = qtw.QVBoxLayout()
        self.side_container.setLayout(self.side_layout)
        self.side_layout.addWidget(
            self.mini_view, alignment=qtc.Qt.AlignCenter)
        self.layout().addWidget(self.side_container)
        self.side_container.setStyleSheet("background-color: white;")

        self.mask_options_container = qtw.QWidget()
        self.mask_options_layout = qtw.QVBoxLayout()
        self.mask_options_container.setLayout(self.mask_options_layout)
        self.side_layout.addWidget(self.mask_options_container)
        self.mask_options_container.setStyleSheet("background-color: #cdf1f9;")

        self.recent_point = []
        self.all_points = []
        self.all_clicks = []
        self.all_lines = []
        self.smooth_points = []

        # Mini scene
        self.mini_scene = MiniGraphicsScene()
        self.mini_scene.setSceneRect(0, 0, Canvas.width()*2, Canvas.height()*2)
        self.mini_view.setScene(self.mini_scene)

        # Put the layer image into the mini canvas
        self.mini_canvas_pixmap = qtg.QPixmap(self.image_path)
        self.mini_canvas_pixmap_item = qtw.QGraphicsPixmapItem(
            self.mini_canvas_pixmap)
        self.mini_add_item = self.mini_scene.addItem(
            self.mini_canvas_pixmap_item)
        # Scale the layer image by 2 (essentially zooming in)
        self.mini_canvas_pixmap_item.setScale(2)

        # Add a cursor indicator image to the mini view
        self.indicator_pixmap = qtg.QPixmap(":/pixel_point.png")
        self.indicator_pixmap_item = qtw.QGraphicsPixmapItem(
            self.indicator_pixmap)
        self.indicator_pixmap_item.setScale(2)
        self.indicator_add_item = self.mini_scene.addItem(
            self.indicator_pixmap_item)
        self.indicator_pixmap_item.setPos(50*2, 50*2)

        self.graphics_scene = CutoutGraphicsScene(self)
        self.graphics_scene.setSceneRect(0, 0, self.imageW, self.imageH)
        self.cutout_view.setScene(self.graphics_scene)
        self.graphics_scene.setView(self.cutout_view)

        self.pen = qtg.QPen(qtg.QBrush(qtg.QColor(0, 0, 0, 255)), 2)

        self.active_image_pixmap = qtg.QPixmap(self.image_path)
        self.active_image_item = CutoutGraphicsItem(self.active_image_pixmap)
        self.add_active_image = self.graphics_scene.addItem(
            self.active_image_item)
        self.cutout_view.updateView()
        self.active_image_item.setEnabled(False)

        self.cutout_view.updateView()
        self.mini_view.setView(0, 0, self)

        self.cutout_tool_title = ToolTitleWidget("Cut-out")
        self.cutout_help_text = """<b>This tool allows you to cut-out a 2D shape from the image. You can draw 
        the path of the 2D shape by clicking on the canvas to plot path points. The 'Join Mask' option will attach 
        the last path point to the first. The 'Smooth Edges' option will produce a cleaner cut-out by curving 
        the edges of the path. You can increase the 'Soften Edges' value to blend the edges of the cut-out 
        when you press 'Mask Image'.</b>"""
        self.cutout_tool_title.setHelpText(self.cutout_help_text)
        self.mask_options_layout.addWidget(self.cutout_tool_title)

        self.soften_edges_widget = qtw.QWidget()
        self.soften_edges_widget_layout = qtw.QFormLayout()
        self.soften_edges_widget.setLayout(self.soften_edges_widget_layout)
        self.feather_factor_input = qtw.QSpinBox()
        self.feather_factor_input.setRange(0, 50)
        self.feather_factor_input.setSingleStep(1)
        self.feather_factor_input.setValue(0)
        self.soften_edges_widget_layout.addRow(
            qtw.QLabel("Soften Edges"), self.feather_factor_input)
        self.mask_options_layout.addWidget(self.soften_edges_widget)

        self.mask_image_button = qtw.QPushButton("Mask Image")
        self.mask_image_button.clicked.connect(
            lambda: self.point_manager.maskClicked(self.feather_factor_input.value()))
        self.mask_options_layout.addWidget(self.mask_image_button)

        self.mask_join_button = qtw.QPushButton("Join Mask")
        self.mask_join_button.clicked.connect(
            lambda: self.point_manager.joinMask())
        self.mask_options_layout.addWidget(self.mask_join_button)

        self.undo_button = qtw.QPushButton("Undo Point")
        self.undo_button.clicked.connect(lambda: self.point_manager.undo())
        self.mask_options_layout.addWidget(self.undo_button)

        self.redo_button = qtw.QPushButton("Redo Point")
        self.redo_button.clicked.connect(lambda: self.point_manager.redo())
        self.mask_options_layout.addWidget(self.redo_button)

        self.smooth_edges_button = qtw.QPushButton("Smooth Edges")
        self.smooth_edges_button.clicked.connect(
            lambda: self.point_manager.smoothEdges())
        self.mask_options_layout.addWidget(self.smooth_edges_button)

        self.show()

    def getPointManager(self):
        return self.point_manager

    def getLineManager(self):
        return self.line_manager


class MainWindow(qtw.QMainWindow):

    def __init__(self):
        super().__init__()

        # Set the main window properties
        self.setWindowTitle('Digital Collage Creator')
        self.window_icon = ":/icon_logo.png"
        self.setWindowIcon(qtg.QIcon(self.window_icon))
        self.setFixedSize(900, 900)
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("background-color: white;")
        self.status_bar.showMessage("Digital Collage Creator...", 4000)

        self.current_stack_widget = None
        self.crop_mode = False

        # Set main container
        # Vertical box that contains toolbar and main section
        self.main_container = qtw.QWidget(self)
        self.main_layout = qtw.QVBoxLayout()
        self.main_container.setLayout(self.main_layout)

        # Creates a button to exit the app
        self.exit_action = qtw.QAction(
            qtg.QIcon(':/icon_quit.png'), 'Exit', self)
        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.triggered.connect(self.quitApp)

        # Save canvas toolbar icon
        self.save_canvas_action = qtw.QAction(
            qtg.QIcon(':/icon_save.png'), 'Save', self)
        self.save_canvas_action.setShortcut('Ctrl+S')
        self.save_canvas_action.triggered.connect(self.saveCanvas)

        # Undo previous action toolbar icon
        self.undo_action = qtw.QAction(
            qtg.QIcon(':/icon_undo.png'), 'Undo', self)
        self.undo_action.setShortcut('Ctrl+Z')
        self.undo_action.triggered.connect(self.undoAction)

        # Redo previous action toolbar icon
        self.redo_action = qtw.QAction(
            qtg.QIcon(':/icon_redo.png'), 'Redo', self)
        self.redo_action.setShortcut('Ctrl+X')
        self.redo_action.triggered.connect(self.redoAction)

        # Home button toolbar icon
        self.home_action = qtw.QAction(
            qtg.QIcon(':/icon_home.png'), 'Home', self)
        self.home_action.triggered.connect(self.homeInfo)

        # Randomise button toolbar icon
        self.random_tool_action = qtw.QAction(
            qtg.QIcon(':/icon_random.png'), 'Randomise', self)
        self.random_tool_action.triggered.connect(self.randomInfo)

        # Create a toolbar and add actions
        self.toolbar = qtw.QToolBar(self)
        self.toolbar.setIconSize(qtc.QSize(25, 25))
        self.toolbar.addAction(self.exit_action)
        self.toolbar.addAction(self.save_canvas_action)
        self.toolbar.addAction(self.undo_action)
        self.toolbar.addAction(self.redo_action)
        self.toolbar.addAction(self.home_action)
        self.toolbar.addAction(self.random_tool_action)

        # Set option icon size
        self.option_icon_size = qtc.QSize(25, 25)
        # Set option tooltip font properties
        qtw.QToolTip.setFont(qtg.QFont('SansSerif', 8))

        # Brightness option
        self.brightness_button = qtw.QPushButton('', self)
        self.brightness_button.setToolTip("Brightness")
        self.brightness_button.setFlat(True)
        self.brightness_button.clicked.connect(self.handleBrightnessButton)
        self.brightness_button.setIcon(qtg.QIcon(':/icon_brightness.png'))
        self.brightness_button.setIconSize(self.option_icon_size)

        # Crop option
        self.crop_button = qtw.QPushButton('', self)
        self.crop_button.setToolTip("Crop")
        self.crop_button.setFlat(True)
        self.crop_button.clicked.connect(self.handleCropButton)
        self.crop_button.setIcon(qtg.QIcon(':/icon_crop.png'))
        self.crop_button.setIconSize(self.option_icon_size)

        # Delete layer option
        self.delete_button = qtw.QPushButton('', self)
        self.delete_button.setToolTip("Delete Layer")
        self.delete_button.setFlat(True)
        self.delete_button.clicked.connect(self.handleDeleteButton)
        self.delete_button.setIcon(qtg.QIcon(':/icon_delete.png'))
        self.delete_button.setIconSize(self.option_icon_size)

        # Black and white filter option
        self.baw_button = qtw.QPushButton('', self)
        self.baw_button.setToolTip("Black & White")
        self.baw_button.setFlat(True)
        self.baw_button.clicked.connect(self.handleBawButton)
        self.baw_button.setIcon(qtg.QIcon(':/icon_blackwhite.png'))
        self.baw_button.setIconSize(self.option_icon_size)

        # Move layer option
        self.move_layer_button = qtw.QPushButton('', self)
        self.move_layer_button.setToolTip("Move Layer")
        self.move_layer_button.setFlat(True)
        self.move_layer_button.clicked.connect(self.handleMoveLayerButton)
        self.move_layer_button.setIcon(qtg.QIcon(':/icon_move.png'))
        self.move_layer_button.setIconSize(self.option_icon_size)

        # Rotate layer option
        self.rotate_layer_button = qtw.QPushButton('', self)
        self.rotate_layer_button.setToolTip("Rotate Layer")
        self.rotate_layer_button.setFlat(True)
        self.rotate_layer_button.clicked.connect(self.handleRotateLayerButton)
        self.rotate_layer_button.setIcon(qtg.QIcon(':/icon_rotate.png'))
        self.rotate_layer_button.setIconSize(self.option_icon_size)

        # Resize layer option
        self.resize_layer_button = qtw.QPushButton('', self)
        self.resize_layer_button.setToolTip("Resize Layer")
        self.resize_layer_button.setFlat(True)
        self.resize_layer_button.clicked.connect(self.handleResizeLayerButton)
        self.resize_layer_button.setIcon(qtg.QIcon(':/icon_resize.png'))
        self.resize_layer_button.setIconSize(self.option_icon_size)

        # Edit rgb option
        self.edit_rgb_button = qtw.QPushButton('', self)
        self.edit_rgb_button.setToolTip("Edit RGB")
        self.edit_rgb_button.setFlat(True)
        self.edit_rgb_button.clicked.connect(self.handleRGBEditButton)
        self.edit_rgb_button.setIcon(qtg.QIcon(':/icon_rgb.png'))
        self.edit_rgb_button.setIconSize(self.option_icon_size)

        # Blur option
        self.blur_button = qtw.QPushButton('', self)
        self.blur_button.setToolTip("Blur")
        self.blur_button.setFlat(True)
        self.blur_button.clicked.connect(self.handleBlurButton)
        self.blur_button.setIcon(qtg.QIcon(':/icon_blur.png'))
        self.blur_button.setIconSize(self.option_icon_size)

        # Text option
        self.add_text_button = qtw.QPushButton('', self)
        self.add_text_button.setToolTip("Add Text")
        self.add_text_button.setFlat(True)
        self.add_text_button.clicked.connect(self.handleAddTextButton)
        self.add_text_button.setIcon(qtg.QIcon(':/icon_text.png'))
        self.add_text_button.setIconSize(self.option_icon_size)

        # Contrast option
        self.contrast_button = qtw.QPushButton('', self)
        self.contrast_button.setToolTip("Contrast")
        self.contrast_button.setFlat(True)
        self.contrast_button.clicked.connect(self.handleContrastButton)
        self.contrast_button.setIcon(qtg.QIcon(':/icon_contrast.png'))
        self.contrast_button.setIconSize(self.option_icon_size)

        # Sharpness option
        self.details_button = qtw.QPushButton('', self)
        self.details_button.setToolTip("Sharpness")
        self.details_button.setFlat(True)
        self.details_button.clicked.connect(self.handleDetailsButton)
        self.details_button.setIcon(qtg.QIcon(':/icon_sharpness.png'))
        self.details_button.setIconSize(self.option_icon_size)

        # Gradient option
        self.gradient_button = qtw.QPushButton('', self)
        self.gradient_button.setToolTip("Gradients")
        self.gradient_button.setFlat(True)
        self.gradient_button.clicked.connect(self.handleGradientButton)
        self.gradient_button.setIcon(qtg.QIcon(':/icon_gradient.png'))
        self.gradient_button.setIconSize(self.option_icon_size)

        # New layer option
        self.add_layer_button = qtw.QPushButton('', self)
        self.add_layer_button.setToolTip("Add Layer")
        self.add_layer_button.setFlat(True)
        self.add_layer_button.clicked.connect(self.handleAddLayerButton)
        self.add_layer_button.setIcon(qtg.QIcon(':/icon_add_layer.png'))
        self.add_layer_button.setIconSize(self.option_icon_size)

        # Alignment option
        self.alignment_button = qtw.QPushButton('', self)
        self.alignment_button.setToolTip("Align Layer")
        self.alignment_button.setFlat(True)
        self.alignment_button.clicked.connect(self.handleAlignLayerButton)
        self.alignment_button.setIcon(qtg.QIcon(':/icon_align.png'))
        self.alignment_button.setIconSize(self.option_icon_size)

        # Cutout layer option
        self.cutout_button = qtw.QPushButton('', self)
        self.cutout_button.setToolTip("Cutout Image")
        self.cutout_button.setFlat(True)
        self.cutout_button.clicked.connect(self.handleCutoutButton)
        self.cutout_button.setIcon(qtg.QIcon(':/icon_cut.png'))
        self.cutout_button.setIconSize(self.option_icon_size)

        # Create the main section
        # Contains the canvas
        # Contains the side container
        self.main_section = qtw.QWidget(self)
        self.main_section_layout = qtw.QHBoxLayout()
        self.main_section.setLayout(self.main_section_layout)

        # Canvas container
        self.canvas_section = qtw.QFrame()
        self.canvas_section.setFixedSize(400, 600)
        self.canvas_section.setFrameShape(qtw.QFrame.Box)
        self.canvas_section.setLineWidth(0)
        # Set container layout
        self.canvas_section_layout = qtw.QVBoxLayout()
        self.canvas_section_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_section_layout.setSpacing(0)
        self.canvas_section.setLayout(self.canvas_section_layout)

        self.canvas = Canvas(800, 1200)

        self.canvas_view = MainGraphicsView()
        self.canvas_view.setFixedSize(400, 600)
        self.canvas_view.setViewportMargins(-2, -2, -2, -2)
        self.canvas_section_layout.addWidget(self.canvas_view)

        self.canvas_scene = MainGraphicsScene()
        self.canvas_scene.setSceneRect(0, 0, Canvas.width(), Canvas.height())
        self.canvas_view.setScene(self.canvas_scene)
        self.canvas_scene.setView(self.canvas_view)

        pen = qtg.QPen(qtg.QBrush(qtg.QColor(0, 0, 0, 255)), 2)

        # Draw a border around the canvas
        top_canvas_border = self.canvas_scene.addLine(
            0, 0, Canvas.width(), 0, pen)
        left_canvas_border = self.canvas_scene.addLine(
            0, 0, 0, Canvas.height(), pen)
        right_canvas_border = self.canvas_scene.addLine(
            Canvas.width(), 0, Canvas.width(), Canvas.height(), pen)
        bottom_canvas_border = self.canvas_scene.addLine(
            0, Canvas.height(), Canvas.width(), Canvas.height(), pen)
        top_canvas_border.setZValue(100)
        left_canvas_border.setZValue(100)
        right_canvas_border.setZValue(100)
        bottom_canvas_border.setZValue(100)
        self.canvas_view.updateView()

        # Create the side section
        # Contains the options section
        # Contains the layers section
        self.side_section = qtw.QWidget()
        self.side_section.setContentsMargins(0, 0, 0, 0)
        self.side_section.setFixedWidth(300)
        self.side_section_layout = qtw.QVBoxLayout()
        self.side_section_layout.setSpacing(2)
        self.side_section_layout.setContentsMargins(0, 0, 0, 0)
        self.side_section.setLayout(self.side_section_layout)

        # Create the options container
        # Contains options like brightness, B&W, blur etc...
        self.options_section = qtw.QWidget()
        self.options_section.setMaximumHeight(200)
        self.options_section_layout = qtw.QGridLayout()
        self.options_section.setLayout(self.options_section_layout)

        # Create the options details container
        self.param_section = qtw.QStackedWidget()
        self.param_section.setFixedWidth(300)
        self.param_section.currentChanged.connect(self.stackChanged)

        # Options info widget
        self.info_widget = qtw.QWidget()
        self.info_form = qtw.QWidget()

        self.info_layout = qtw.QVBoxLayout()
        self.info_form_layout = qtw.QFormLayout()

        self.info_label = qtw.QLabel("<b>Home</b>")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(qtc.Qt.AlignCenter)

        self.info_help_label = qtw.QLabel("Use the layer manager below to add image layers and select the active layer. Select a tool from the toolbox above to activate it. " +
                                          "You can hover over icons to see what they do and within each tool you can hover over the question mark to reveal more information. Remember, you can view this information " +
                                          "again by pressing the home icon in the toolbar!")
        self.info_help_label.setWordWrap(True)
        self.info_help_label.setAlignment(qtc.Qt.AlignCenter)

        self.info_form_layout.addRow(self.info_label)
        self.info_form_layout.addRow(self.info_help_label)

        self.info_widget.setLayout(self.info_layout)
        self.info_form.setLayout(self.info_form_layout)
        self.info_layout.addWidget(
            self.info_form, alignment=qtc.Qt.AlignCenter)

        # Move image layer tool info
        self.move_param_widget = qtw.QWidget()
        self.move_form = qtw.QWidget()

        self.move_form_layout = qtw.QFormLayout()
        self.move_param_layout = qtw.QVBoxLayout()

        self.move_instructions_label = qtw.QLabel(
            "Select an active layer to be able to grab and drag the image around the canvas")
        self.move_instructions_label.setWordWrap(True)
        self.move_instructions_label.setAlignment(qtc.Qt.AlignCenter)

        self.move_tooltip_text = """<b>The Move tool allows you to grab and drag the image of the active layer into a new position on the canvas. Select an active layer from
        the layer manager below.</b>"""
        self.move_tool_title = ToolTitleWidget("Move")
        self.move_tool_title.setHelpText(self.move_tooltip_text)

        self.move_form_layout.addRow(self.move_instructions_label)

        self.move_form.setLayout(self.move_form_layout)
        self.move_param_widget.setLayout(self.move_param_layout)
        self.a = qtc.Qt.AlignHCenter | qtc.Qt.AlignTop
        self.move_param_layout.addWidget(
            self.move_tool_title, alignment=qtc.Qt.AlignHCenter | qtc.Qt.AlignBottom)
        self.move_param_layout.addWidget(
            self.move_form, alignment=qtc.Qt.AlignHCenter | qtc.Qt.AlignTop)

        # Resize image layer tool info
        self.resize_factor_widget = qtw.QWidget()
        self.resize_factor_layout = qtw.QVBoxLayout()

        self.resize_form = qtw.QWidget()
        self.resize_form_layout = qtw.QFormLayout()

        self.resize_title_label = ToolTitleWidget("Resize")
        self.resize_tooltip_text = """<b>The Resize tool allows you increase or decrease the size of the active layer. The icon 
        will appear on the canvas, at the top right corner of the active layer. Grab this icon and move you mouse closer/further 
        away from the center of the image to scale the size up or down.</b>"""
        self.resize_title_label.setHelpText(self.resize_tooltip_text)

        self.resize_instructions_label = qtw.QLabel(
            "Select an active layer to activate the resize icon on the canvas. Grab the icon and move the mouse the scale the image.")
        self.resize_instructions_label.setWordWrap(True)
        self.resize_instructions_label.setAlignment(qtc.Qt.AlignCenter)

        self.resize_form_layout.addRow(self.resize_title_label)
        self.resize_form_layout.addRow(self.resize_instructions_label)

        self.resize_form.setLayout(self.resize_form_layout)
        self.resize_factor_widget.setLayout(self.resize_factor_layout)
        self.resize_factor_layout.addWidget(
            self.resize_form, alignment=qtc.Qt.AlignCenter)

        # Black and white tool info
        self.baw_widget = qtw.QWidget()
        self.baw_form = qtw.QWidget()

        self.baw_form_layout = qtw.QFormLayout()
        self.baw_layout = qtw.QVBoxLayout()

        self.baw_label = ToolTitleWidget("Black & White")
        self.baw_tooltip_text = """<b>The Black & White tool applies a simple black & white filter onto the active layer.</b>"""
        self.baw_label.setHelpText(self.baw_tooltip_text)

        self.baw_submit = qtw.QPushButton("Apply Filter")
        self.baw_submit.clicked.connect(self.submitBaw)

        self.baw_form_layout.addRow(self.baw_label)
        self.baw_form_layout.addRow(self.baw_submit)

        self.baw_form.setLayout(self.baw_form_layout)
        self.baw_widget.setLayout(self.baw_layout)
        self.baw_layout.addWidget(self.baw_form, alignment=qtc.Qt.AlignCenter)

        # Rotate by angle tool info
        self.rotate_widget = qtw.QWidget()
        self.rotate_form = qtw.QWidget()

        self.rotate_layout = qtw.QVBoxLayout()
        self.rotate_form_layout = qtw.QFormLayout()

        self.rotate_label = ToolTitleWidget("Rotate")
        self.rotate_tooltip_text = """<b>The Rotate tool allows you to alter the angle of the active layer. Simply grab the icon 
        that appears in the top right corner of the active layer and move the mouse to rotate the layer either clockwise or 
        anti-clockwise.</b>"""
        self.rotate_label.setHelpText(self.rotate_tooltip_text)

        self.rotate_instructions_label = qtw.QLabel(
            "Select an active layer to activate the rotate icon on the canvas. Grab the icon and move the mouse around the image to rotate.")
        self.rotate_instructions_label.setWordWrap(True)
        self.rotate_instructions_label.setAlignment(qtc.Qt.AlignCenter)

        self.rotate_form_layout.addRow(self.rotate_label)
        self.rotate_form_layout.addRow(self.rotate_instructions_label)

        self.rotate_form.setLayout(self.rotate_form_layout)
        self.rotate_widget.setLayout(self.rotate_layout)
        self.rotate_layout.addWidget(
            self.rotate_form, alignment=qtc.Qt.AlignCenter)

        # Brightness tool info
        self.brightness_widget = qtw.QWidget()
        self.brightness_form = qtw.QWidget()

        self.brightness_layout = qtw.QVBoxLayout()
        self.brightness_form_layout = qtw.QFormLayout()

        self.brightness_label = ToolTitleWidget("Brightness")
        self.brightness_tooltip_text = """<b>The Brightness tool allows you to either decrease or increase the brightness of the active layer. 
        Setting the value to 0% will produce a fully black image.</b>"""
        self.brightness_label.setHelpText(self.brightness_tooltip_text)

        self.brightness_param_submit = qtw.QPushButton("Alter")
        self.brightness_param_submit.clicked.connect(self.brightnessSubmit)

        self.brightness_param_factor = qtw.QSlider()
        self.brightness_param_factor.setOrientation(qtc.Qt.Horizontal)
        self.brightness_param_factor.setRange(-100, 100)
        self.brightness_param_factor.setSingleStep(1)
        self.brightness_param_factor.setValue(0)
        self.brightness_factor_label = qtw.QLabel()
        self.brightness_factor_label.setText(
            str(self.brightness_param_factor.value()) + "%")
        self.brightness_factor_label.setAlignment(qtc.Qt.AlignCenter)
        self.brightness_param_factor.valueChanged.connect(
            self.updateBrightnessLabel)

        self.brightness_form_layout.addRow(self.brightness_label)
        self.brightness_form_layout.addRow(self.brightness_param_factor)
        self.brightness_form_layout.addRow(self.brightness_factor_label)
        self.brightness_form_layout.addRow(self.brightness_param_submit)

        self.brightness_form.setLayout(self.brightness_form_layout)
        self.brightness_widget.setLayout(self.brightness_layout)
        self.brightness_layout.addWidget(
            self.brightness_form, alignment=qtc.Qt.AlignCenter)

        # Contrast tool info
        self.contrast_widget = qtw.QWidget()
        self.contrast_form = qtw.QWidget()

        self.contrast_form_layout = qtw.QFormLayout()
        self.contrast_layout = qtw.QVBoxLayout()

        self.contrast_label = ToolTitleWidget("<b>Contrast</b>")
        self.contrast_tooltip_text = """<b>The Contrast tool allows you to decrease or increase the contrast of the 
        active layer. Contrast is the range of brightness within the image.</b>"""
        self.contrast_label.setHelpText(self.contrast_tooltip_text)

        self.contrast_param_submit = qtw.QPushButton("Alter")
        self.contrast_param_submit.clicked.connect(self.contrastSubmit)

        self.contrast_param_factor = qtw.QSlider()
        self.contrast_param_factor.setOrientation(qtc.Qt.Horizontal)
        self.contrast_param_factor.setRange(-100, 100)
        self.contrast_param_factor.setSingleStep(1)
        self.contrast_param_factor.setValue(0)
        self.contrast_factor_label = qtw.QLabel()
        self.contrast_factor_label.setText(
            str(self.contrast_param_factor.value()) + "%")
        self.contrast_factor_label.setAlignment(qtc.Qt.AlignCenter)
        self.contrast_param_factor.valueChanged.connect(
            self.updateContrastLabel)

        self.contrast_form_layout.addRow(self.contrast_label)
        self.contrast_form_layout.addRow(self.contrast_param_factor)
        self.contrast_form_layout.addRow(self.contrast_factor_label)
        self.contrast_form_layout.addRow(self.contrast_param_submit)

        self.contrast_form.setLayout(self.contrast_form_layout)
        self.contrast_widget.setLayout(self.contrast_layout)
        self.contrast_layout.addWidget(
            self.contrast_form, alignment=qtc.Qt.AlignCenter)

        # Crop tool info
        self.crop_widget = qtw.QWidget()
        self.crop_form = qtw.QWidget()

        self.crop_layout = qtw.QVBoxLayout()
        self.crop_form_layout = qtw.QFormLayout()

        self.crop_label = ToolTitleWidget("<b>Crop</b>")
        self.crop_tooltip_text = """<b>The crop tool allows you to draw a rectangle on the canvas and crop 
        (keep) that section of the active layer. Draw the rectangle by clicking the mouse on the canvas and 
        dragging."""
        self.crop_label.setHelpText(self.crop_tooltip_text)

        self.crop_instructions_label = qtw.QLabel(
            "Click and drag on the canvas to draw a box around the area of the image to keep")
        self.crop_instructions_label.setWordWrap(True)
        self.crop_instructions_label.setAlignment(qtc.Qt.AlignCenter)

        self.crop_param_submit = qtw.QPushButton("Crop Image")

        self.crop_param_submit.clicked.connect(self.cropSubmit)
        self.crop_clear_button = qtw.QPushButton("Clear Box")
        self.crop_clear_button.clicked.connect(self.canvas_scene.deleteCropBox)

        self.crop_form_layout.addRow(self.crop_label)
        self.crop_form_layout.addRow(self.crop_instructions_label)
        self.crop_form_layout.addRow(self.crop_param_submit)
        self.crop_form_layout.addRow(self.crop_clear_button)

        self.crop_form.setLayout(self.crop_form_layout)
        self.crop_widget.setLayout(self.crop_layout)
        self.crop_layout.addWidget(
            self.crop_form, alignment=qtc.Qt.AlignCenter)

        # Blur tool info
        self.blur_widget = qtw.QWidget()
        self.blur_widget_layout = qtw.QVBoxLayout()

        self.blur_form = qtw.QWidget()

        self.blur_layout = qtw.QFormLayout()

        self.blur_label = ToolTitleWidget("Blur")
        self.blur_help_text = """<b>The Blur tool allows you to apply a blur effect over the active layer. 
        The effect can be turned off.</b>"""
        self.blur_label.setHelpText(self.blur_help_text)

        self.blur_param_submit = qtw.QPushButton("Toggle Blur")
        self.blur_param_submit.clicked.connect(self.blurSubmit)
        self.blur_layout.addRow(self.blur_label)
        self.blur_layout.addRow(self.blur_param_submit)

        self.blur_form.setLayout(self.blur_layout)
        self.blur_widget.setLayout(self.blur_widget_layout)
        self.blur_widget_layout.addWidget(
            self.blur_form, alignment=qtc.Qt.AlignCenter)

        # Add text tool info
        self.text_widget = qtw.QWidget()
        self.text_form = qtw.QWidget()

        self.text_form_layout = qtw.QFormLayout()
        self.text_layout = qtw.QVBoxLayout()

        self.text_label = ToolTitleWidget("Add Text")
        self.text_help_text = """<b>The Text tool allows you to add a piece of text as a new layer 
        onto the canvas. Simply enter the text you want to add, input the size, font and colour and 
        press the 'Add Text' button.</b>"""
        self.text_label.setHelpText(self.text_help_text)

        self.text_param_submit = qtw.QPushButton("Add Text")
        self.text_param_submit.clicked.connect(self.textSubmit)

        self.text_param_text = qtw.QLineEdit()
        self.text_param_font = qtw.QLineEdit()
        self.text_param_size = qtw.QLineEdit()
        self.text_param_colour = qtw.QLineEdit()
        self.text_param_font.setText("LemonMilk.otf")
        self.text_param_size.setText("100")
        self.text_param_colour.setText("000000")

        self.text_form_layout.addRow(self.text_label)
        self.text_form_layout.addRow("Text:", self.text_param_text)
        self.text_form_layout.addRow("Font:", self.text_param_font)
        self.text_form_layout.addRow("Size (px):", self.text_param_size)
        self.text_form_layout.addRow("Colour (hex): #", self.text_param_colour)
        self.text_form_layout.addRow(self.text_param_submit)

        self.text_form.setLayout(self.text_form_layout)
        self.text_widget.setLayout(self.text_layout)
        self.text_layout.addWidget(
            self.text_form, alignment=qtc.Qt.AlignCenter)

        # Save tool info
        self.save_widget = qtw.QWidget()
        self.save_form = qtw.QWidget()

        self.save_form_layout = qtw.QFormLayout()
        self.save_layout = qtw.QVBoxLayout()

        self.save_label = qtw.QLabel("<b>Save/Open Collage</b>")
        self.save_label.setWordWrap(True)
        self.save_label.setAlignment(qtc.Qt.AlignCenter)

        self.save_param_submit = qtw.QPushButton("Save As Image")
        self.save_param_submit.clicked.connect(self.saveSubmit)

        self.save_project_submit = qtw.QPushButton("Save As Project")
        self.save_project_submit.clicked.connect(self.saveProjectSubmit)

        self.save_project_open = qtw.QPushButton("Open Project")
        self.save_project_open.clicked.connect(self.openProjectSubmit)

        self.save_form_layout.addRow(self.save_label)
        self.save_form_layout.addRow(self.save_param_submit)
        self.save_form_layout.addRow(self.save_project_submit)
        self.save_form_layout.addRow(self.save_project_open)

        self.save_form.setLayout(self.save_form_layout)
        self.save_widget.setLayout(self.save_layout)
        self.save_layout.addWidget(
            self.save_form, alignment=qtc.Qt.AlignCenter)

        # Delete tool info
        self.delete_widget = qtw.QWidget()
        self.delete_form = qtw.QWidget()

        self.delete_form_layout = qtw.QFormLayout()
        self.delete_layout = qtw.QVBoxLayout()

        self.delete_label = ToolTitleWidget("Delete")
        self.delete_help_text = """<b>The Delete tool removes the active layer from the canvas. This 
        can be reversed by using the Undo tool in the top toolbar.</b>"""
        self.delete_label.setHelpText(self.delete_help_text)

        self.delete_submit = qtw.QPushButton("Delete")
        self.delete_submit.clicked.connect(self.deleteSubmit)

        self.delete_form_layout.addRow(self.delete_label)
        self.delete_form_layout.addRow(self.delete_submit)

        self.delete_form.setLayout(self.delete_form_layout)
        self.delete_widget.setLayout(self.delete_layout)
        self.delete_layout.addWidget(
            self.delete_form, alignment=qtc.Qt.AlignCenter)

        # Sharpness tool info
        self.details_widget = qtw.QWidget()
        self.details_form = qtw.QWidget()

        self.details_form_layout = qtw.QFormLayout()
        self.details_layout = qtw.QVBoxLayout()

        self.details_label = ToolTitleWidget("Sharpness")
        self.details_help_text = """<b>The Sharpness tool allows you to decrease or increase the 
        quality of the active layer image.</b>"""
        self.details_label.setHelpText(self.details_help_text)

        self.details_submit = qtw.QPushButton("Alter")

        self.details_param_factor = qtw.QSlider()
        self.details_param_factor.setOrientation(qtc.Qt.Horizontal)
        self.details_param_factor.setRange(-100, 100)
        self.details_param_factor.setSingleStep(1)
        self.details_param_factor.setValue(0)
        self.details_factor_label = qtw.QLabel()
        self.details_factor_label.setAlignment(qtc.Qt.AlignCenter)
        self.details_factor_label.setText(
            str(self.details_param_factor.value()) + "%")
        self.details_param_factor.valueChanged.connect(self.updateDetailsLabel)

        self.details_submit.clicked.connect(self.detailsSubmit)

        self.details_form_layout.addRow(self.details_label)
        self.details_form_layout.addRow(self.details_param_factor)
        self.details_form_layout.addRow(self.details_factor_label)
        self.details_form_layout.addRow(self.details_submit)

        self.details_form.setLayout(self.details_form_layout)
        self.details_widget.setLayout(self.details_layout)
        self.details_layout.addWidget(
            self.details_form, alignment=qtc.Qt.AlignCenter)

        # RGB tool info
        self.rgb_widget = qtw.QWidget()
        self.rgb_form = qtw.QWidget()
        self.rgb_sliders = qtw.QWidget()

        self.rgb_sliders_layout = qtw.QGridLayout()
        self.rgb_form_layout = qtw.QFormLayout()
        self.rgb_layout = qtw.QVBoxLayout()

        self.rgb_label = ToolTitleWidget("Colour Editing")
        self.rgb_help_text = """<b>Each part of an image is made up of colours, these colours 
        have a red, green and blue value which are combined to create the colours you see. The
        Colour Editing tool allows you to reduce or increase the intensity of the red, green and
        blue values within the image layer.</b>"""
        self.rgb_label.setHelpText(self.rgb_help_text)

        self.rgb_r_factor = qtw.QSlider()
        self.rgb_r_factor.setOrientation(qtc.Qt.Horizontal)
        self.rgb_r_factor.setRange(-100, 100)
        self.rgb_r_factor.setSingleStep(1)
        self.rgb_r_factor.setValue(0)
        self.rgb_r_label = qtw.QLabel()
        self.rgb_r_label.setMinimumWidth(40)
        self.rgb_r_label.setText(str(self.rgb_r_factor.value()) + "%")
        self.rgb_r_factor.valueChanged.connect(self.updateRGBRLabel)

        self.rgb_g_factor = qtw.QSlider()
        self.rgb_g_factor.setOrientation(qtc.Qt.Horizontal)
        self.rgb_g_factor.setRange(-100, 100)
        self.rgb_g_factor.setSingleStep(1)
        self.rgb_g_factor.setValue(0)
        self.rgb_g_label = qtw.QLabel()
        self.rgb_g_label.setMinimumWidth(40)
        self.rgb_g_label.setText(str(self.rgb_g_factor.value()) + "%")
        self.rgb_g_factor.valueChanged.connect(self.updateRGBGLabel)

        self.rgb_b_factor = qtw.QSlider()
        self.rgb_b_factor.setOrientation(qtc.Qt.Horizontal)
        self.rgb_b_factor.setRange(-100, 100)
        self.rgb_b_factor.setSingleStep(1)
        self.rgb_b_factor.setValue(0)
        self.rgb_b_label = qtw.QLabel()
        self.rgb_b_label.setMinimumWidth(40)
        self.rgb_b_label.setText(str(self.rgb_b_factor.value()) + "%")
        self.rgb_b_factor.valueChanged.connect(self.updateRGBBLabel)

        self.rgb_red_text = qtw.QLabel("<b>Red</b>")
        self.rgb_green_text = qtw.QLabel("<b>Green</b>")
        self.rgb_blue_text = qtw.QLabel("<b>Blue</b>")

        self.rgb_submit = qtw.QPushButton("Alter")
        self.rgb_submit.clicked.connect(self.rgbSubmit)

        self.rgb_form_layout.addRow(self.rgb_label)
        self.rgb_sliders_layout.addWidget(self.rgb_red_text, 0, 0)
        self.rgb_sliders_layout.addWidget(self.rgb_r_label, 0, 1)
        self.rgb_sliders_layout.addWidget(self.rgb_r_factor, 0, 2)
        self.rgb_sliders_layout.addWidget(self.rgb_green_text, 1, 0)
        self.rgb_sliders_layout.addWidget(self.rgb_g_label, 1, 1)
        self.rgb_sliders_layout.addWidget(self.rgb_g_factor, 1, 2)
        self.rgb_sliders_layout.addWidget(self.rgb_blue_text, 2, 0)
        self.rgb_sliders_layout.addWidget(self.rgb_b_label, 2, 1)
        self.rgb_sliders_layout.addWidget(self.rgb_b_factor, 2, 2)
        self.rgb_form_layout.addRow(self.rgb_sliders)
        self.rgb_form_layout.addRow(self.rgb_submit)

        self.rgb_sliders.setLayout(self.rgb_sliders_layout)
        self.rgb_form.setLayout(self.rgb_form_layout)
        self.rgb_widget.setLayout(self.rgb_layout)
        self.rgb_layout.addWidget(self.rgb_form, alignment=qtc.Qt.AlignCenter)

        # Gradient tool info
        self.gradient_widget = qtw.QWidget()
        self.gradient_container = qtw.QVBoxLayout()

        self.gradient_image = qtw.QWidget()
        self.gradient_layout = qtw.QFormLayout()

        self.gradient_label = ToolTitleWidget("Gradient Backgrounds")
        self.gradient_help_text = """<b>The 'Get Active Layer Colours' button will provide 
        you with a colour palette containing the 7 most dominant colours within the active
        layer image. These colours can then be selected from to create a gradient background (a smooth 
        blend of colours) by clicking the 'Generate Gradient' button.</b>"""
        self.gradient_label.setHelpText(self.gradient_help_text)

        self.gradient_submit = qtw.QPushButton("Get Active Layer Colours")
        self.gradient_submit.clicked.connect(self.gradientSubmit)

        self.gradient_grid = qtw.QGridLayout()

        self.gradient_active_1 = qtw.QLabel()
        self.gradient_active_1.setFixedSize(30, 30)
        self.gradient_active_2 = qtw.QLabel()
        self.gradient_active_2.setFixedSize(30, 30)

        self.gradient_generate = qtw.QPushButton("Generate Gradient")
        self.gradient_generate.clicked.connect(self.gradientGenerate)

        self.gradient_choices = qtw.QWidget()
        self.gradient_choices_layout = qtw.QGridLayout()
        self.gradient_choices_layout.addWidget(self.gradient_active_1, 0, 0)
        self.gradient_choices_layout.addWidget(self.gradient_active_2, 0, 1)
        self.gradient_choices_layout.addWidget(
            self.gradient_generate, 1, 0, 1, 2, alignment=qtc.Qt.AlignCenter)

        GradientManager.setActive1Widget(self.gradient_active_1)
        GradientManager.setActive2Widget(self.gradient_active_2)

        self.gradient_layout.addRow(self.gradient_label)
        self.gradient_layout.addRow(self.gradient_submit)
        self.gradient_layout.addRow(self.gradient_grid)
        self.gradientSubmit()

        self.gradient_container.addWidget(self.gradient_image)
        self.gradient_container.addWidget(self.gradient_choices)

        self.gradient_widget.setLayout(self.gradient_container)
        self.gradient_image.setLayout(self.gradient_layout)
        self.gradient_choices.setLayout(self.gradient_choices_layout)

        # Cutout image tool info
        self.cutout_widget = qtw.QWidget()
        self.cutout_form = qtw.QWidget()

        self.cutout_form_layout = qtw.QFormLayout()
        self.cutout_layout = qtw.QVBoxLayout()

        self.cutout_label = qtw.QLabel("<b>Cutout Layer</b>")
        self.cutout_label.setWordWrap(True)
        self.cutout_label.setAlignment(qtc.Qt.AlignCenter)

        self.cutout_instructions_label = qtw.QLabel("Pressing the 'Cut' button will open a new window containing the active layer image " +
                                                    "where you can click the image to plot path points. Each path point will automatically join together. Draw the path around around " +
                                                    "the section of the image you would like to cut out.")
        self.cutout_instructions_label.setWordWrap(True)
        self.cutout_instructions_label.setAlignment(qtc.Qt.AlignCenter)

        self.cutout_submit = qtw.QPushButton("Cut")
        self.cutout_submit.clicked.connect(self.cutoutSubmit)

        self.cutout_form_layout.addRow(self.cutout_label)
        self.cutout_form_layout.addRow(self.cutout_instructions_label)
        self.cutout_form_layout.addRow(self.cutout_submit)

        self.cutout_form.setLayout(self.cutout_form_layout)
        self.cutout_widget.setLayout(self.cutout_layout)
        self.cutout_layout.addWidget(
            self.cutout_form, alignment=qtc.Qt.AlignCenter)

        # Randomise tool info
        self.randomise_container = qtw.QWidget()
        self.randomise_container_layout = qtw.QVBoxLayout()
        self.randomise_container.setLayout(self.randomise_container_layout)

        self.randomise_content = qtw.QWidget()
        self.randomise_content_layout = qtw.QVBoxLayout()
        self.randomise_content.setLayout(self.randomise_content_layout)

        self.randomise_scroll_area = qtw.QScrollArea()
        self.randomise_scroll_area.setFrameShape(qtw.QFrame.NoFrame)
        self.randomise_scroll_area.setVerticalScrollBarPolicy(
            qtc.Qt.ScrollBarAlwaysOn)
        self.randomise_scroll_area.setHorizontalScrollBarPolicy(
            qtc.Qt.ScrollBarAlwaysOff)
        self.randomise_scroll_area.setStyleSheet(
            "QScrollBar { background-color: #b3e9f6; width:15px;}")
        self.randomise_scroll_area.setWidgetResizable(True)

        self.randomise_title = ToolTitleWidget("Randomise")
        self.randomise_help_text = """<b>The Randomise tool allows you to randomise the properties 
        of a layer. You are able to select which properties of the layer to lock in place, meaning 
        that they will not be changed during the randomisation process.</b>"""
        self.randomise_title.setHelpText(self.randomise_help_text)

        self.randomise_content_layout.addWidget(self.randomise_title)

        self.randomise_scroll_area.setWidget(self.randomise_content)
        self.randomise_container_layout.addWidget(self.randomise_scroll_area)

        # Alignment tool info
        self.alignment_widget = qtw.QWidget()
        self.alignment_layout = qtw.QGridLayout()
        self.alignment_widget.setLayout(self.alignment_layout)

        self.align_v_top_button = qtw.QPushButton('', self)
        self.align_v_top_button.setToolTip("Align Vertical Top")
        self.align_v_top_button.setFlat(True)
        self.align_v_top_button.clicked.connect(self.handleAlignVTop)
        self.align_v_top_button.setIcon(qtg.QIcon(':/icon_vertical_top.png'))
        self.align_v_top_button.setIconSize(self.option_icon_size)

        self.align_v_center_button = qtw.QPushButton('', self)
        self.align_v_center_button.setToolTip("Align Vertical Center")
        self.align_v_center_button.setFlat(True)
        self.align_v_center_button.clicked.connect(self.handleAlignVCenter)
        self.align_v_center_button.setIcon(
            qtg.QIcon(':/icon_vertical_center.png'))
        self.align_v_center_button.setIconSize(self.option_icon_size)

        self.align_v_bottom_button = qtw.QPushButton('', self)
        self.align_v_bottom_button.setToolTip("Align Vertical Top")
        self.align_v_bottom_button.setFlat(True)
        self.align_v_bottom_button.clicked.connect(self.handleAlignVBottom)
        self.align_v_bottom_button.setIcon(
            qtg.QIcon(':/icon_vertical_bottom.png'))
        self.align_v_bottom_button.setIconSize(self.option_icon_size)

        self.align_h_left_button = qtw.QPushButton('', self)
        self.align_h_left_button.setToolTip("Align Horizontal Left")
        self.align_h_left_button.setFlat(True)
        self.align_h_left_button.clicked.connect(self.handleAlignHLeft)
        self.align_h_left_button.setIcon(
            qtg.QIcon(':/icon_horizontal_left.png'))
        self.align_h_left_button.setIconSize(self.option_icon_size)

        self.align_h_center_button = qtw.QPushButton('', self)
        self.align_h_center_button.setToolTip("Align Horizontal Center")
        self.align_h_center_button.setFlat(True)
        self.align_h_center_button.clicked.connect(self.handleAlignHCenter)
        self.align_h_center_button.setIcon(
            qtg.QIcon(':/icon_horizontal_center.png'))
        self.align_h_center_button.setIconSize(self.option_icon_size)

        self.align_h_right_button = qtw.QPushButton('', self)
        self.align_h_right_button.setToolTip("Align Horizontal Right")
        self.align_h_right_button.setFlat(True)
        self.align_h_right_button.clicked.connect(self.handleAlignHRight)
        self.align_h_right_button.setIcon(
            qtg.QIcon(':/icon_horizontal_right.png'))
        self.align_h_right_button.setIconSize(self.option_icon_size)

        self.alignment_layout.addWidget(self.align_v_top_button, 0, 0)
        self.alignment_layout.addWidget(self.align_v_center_button, 0, 1)
        self.alignment_layout.addWidget(self.align_v_bottom_button, 0, 2)
        self.alignment_layout.addWidget(self.align_h_left_button, 1, 0)
        self.alignment_layout.addWidget(self.align_h_center_button, 1, 1)
        self.alignment_layout.addWidget(self.align_h_right_button, 1, 2)

        self.param_section.addWidget(self.move_param_widget)
        self.param_section.addWidget(self.resize_factor_widget)
        self.param_section.addWidget(self.info_widget)
        self.param_section.addWidget(self.baw_widget)
        self.param_section.addWidget(self.rotate_widget)
        self.param_section.addWidget(self.brightness_widget)
        self.param_section.addWidget(self.crop_widget)
        self.param_section.addWidget(self.blur_widget)
        self.param_section.addWidget(self.text_widget)
        self.param_section.addWidget(self.save_widget)
        self.param_section.addWidget(self.delete_widget)
        self.param_section.addWidget(self.contrast_widget)
        self.param_section.addWidget(self.details_widget)
        self.param_section.addWidget(self.rgb_widget)
        self.param_section.addWidget(self.alignment_widget)
        self.param_section.addWidget(self.cutout_widget)
        self.param_section.addWidget(self.gradient_widget)
        self.param_section.addWidget(self.randomise_container)

        self.param_section.setCurrentWidget(self.info_widget)

        # Create the layers section container
        # Contains each layer widget
        self.layers_section = qtw.QWidget()
        self.layers_section_layout = qtw.QVBoxLayout()
        self.layers_section.setLayout(self.layers_section_layout)

        # Create the scrolling layer container
        # Scroll Area which contains the widgets, set as the centralWidget
        self.layer_manager_scroll_area = qtw.QScrollArea()
        # Widget that contains the collection of Vertical Box
        self.layer_manager_container = qtw.QWidget()
        # The Vertical Box that contains the Horizontal Boxes of labels and buttons
        self.layer_manager = qtw.QVBoxLayout()
        self.layer_manager_container.setLayout(self.layer_manager)
        self.layer_manager_scroll_area.setMinimumHeight(270)
        # Scroll Area Properties
        self.layer_manager_scroll_area.setFrameShape(qtw.QFrame.NoFrame)
        self.layer_manager_scroll_area.setVerticalScrollBarPolicy(
            qtc.Qt.ScrollBarAlwaysOn)
        self.layer_manager_scroll_area.setHorizontalScrollBarPolicy(
            qtc.Qt.ScrollBarAlwaysOff)
        self.layer_manager_scroll_area.setStyleSheet(
            "QScrollBar { background-color: #b3e9f6; width:15px;}")
        self.layer_manager_scroll_area.setWidgetResizable(True)
        self.layer_manager_scroll_area.setWidget(self.layer_manager_container)
        self.layers_section_layout.addWidget(self.layer_manager_scroll_area)

        # Set main widget as main container
        self.setCentralWidget(self.main_container)

        # Populate window with widget
        self.main_layout.addWidget(self.toolbar)

        self.main_layout.addWidget(self.main_section)

        self.main_section_layout.addWidget(self.canvas_section)
        self.main_section_layout.addWidget(self.side_section)

        self.side_section_layout.addWidget(self.options_section)
        self.side_section_layout.addWidget(self.param_section)
        self.side_section_layout.addWidget(self.layers_section)

        self.options_section_layout.addWidget(self.brightness_button, 3, 0)
        self.options_section_layout.addWidget(self.delete_button, 0, 1)
        self.options_section_layout.addWidget(self.baw_button, 2, 0)
        self.options_section_layout.addWidget(self.crop_button, 0, 0)
        self.options_section_layout.addWidget(self.rotate_layer_button, 1, 1)
        self.options_section_layout.addWidget(self.move_layer_button, 1, 2)
        self.options_section_layout.addWidget(self.resize_layer_button, 1, 0)
        self.options_section_layout.addWidget(self.edit_rgb_button, 2, 1)
        self.options_section_layout.addWidget(self.blur_button, 3, 1)
        self.options_section_layout.addWidget(self.add_text_button, 4, 2)
        self.options_section_layout.addWidget(self.contrast_button, 2, 2)
        self.options_section_layout.addWidget(self.details_button, 3, 2)
        self.options_section_layout.addWidget(self.gradient_button, 4, 0)
        self.options_section_layout.addWidget(self.alignment_button, 4, 1)
        self.options_section_layout.addWidget(self.cutout_button, 0, 2)

        # Layer manager title and tooltip
        self.layer_manager_title = ToolTitleWidget("Layer Manager")
        self.layer_manager_help_text = """<b>The Layer Manager allows you to view and control the collage layers. 
        To add a new layer to the image press the '+' icon below. You will see the information of each layer is 
        shown in this box, each layer has 4 icon options. The up and down arrow icons allows you to move the layer 
        above or below other layers. The eye icon allows you turn the visibility of the layer on and off. The switch 
        icon allows you to enable or disable a layer as the active layer. To use tools on a layer you will have to 
        activate the layer by pressing that switch.</b>"""
        self.layer_manager_title.setHelpText(self.layer_manager_help_text)
        self.layer_manager.addWidget(self.layer_manager_title)
        self.layer_manager.addWidget(self.add_layer_button)

        # Set background colours of widgets
        self.setStyleSheet("background-color: #cdf1f9;")
        self.toolbar.setStyleSheet("background-color: white;")
        self.side_section.setStyleSheet("background-color: white;")
        self.options_section.setStyleSheet("background-color: #cdf1f9;")
        self.param_section.setStyleSheet("background-color: #cdf1f9;")
        self.layers_section.setStyleSheet("background-color: #cdf1f9;")
        self.layer_manager_container.setStyleSheet(
            "background-color: #cdf1f9;")
        self.canvas_section.setStyleSheet("background-color: white;")
        self.main_section.setStyleSheet("background-color: white;")

        self.show()

    def updateBrightnessLabel(self, value):
        self.brightness_factor_label.setText(str(value) + "%")

    def updateContrastLabel(self, value):
        self.contrast_factor_label.setText(str(value) + "%")

    def updateDetailsLabel(self, value):
        self.details_factor_label.setText(str(value) + "%")

    def updateRGBRLabel(self, value):
        self.rgb_r_label.setText(str(round(value)) + "%")

    def updateRGBGLabel(self, value):
        self.rgb_g_label.setText(str(round(value)) + "%")

    def updateRGBBLabel(self, value):
        self.rgb_b_label.setText(str(round(value)) + "%")

    def updateRGBLabels(self, r, g, b):
        self.updateRGBRLabel(r)
        self.updateRGBGLabel(g)
        self.updateRGBBLabel(b)

    def updateRGBRSlider(self, value):
        self.rgb_r_factor.setValue(round(value))

    def updateRGBGSlider(self, value):
        self.rgb_g_factor.setValue(round(value))

    def updateRGBBSlider(self, value):
        self.rgb_b_factor.setValue(round(value))

    def updateRGBSliders(self, r, g, b):
        self.updateRGBRSlider(r)
        self.updateRGBGSlider(g)
        self.updateRGBBSlider(b)

    def addLayerWidget(self, layer):
        self.layer_manager.addWidget(layer.getLayerWidget())

    def addRandomiseWidget(self, layer):
        self.randomise_content_layout.addWidget(layer.getRandomiseWidget())

    def stackChanged(self, index):
        self.current_stack_widget = index
        self.checkStackOption()

    def checkStackOption(self):
        # Called when the tool info stack is changed
        index = self.current_stack_widget
        self.currentActiveLayer = LayerManager.getActiveLayer()

        if (index == 6):
            # Crop mode is activated
            mw.status_bar.showMessage("Cropping mode...", 3000)
            self.canvas_scene.setCropMode(True)
        else:
            # Crop mode is not activated
            self.canvas_scene.setCropMode(False)
            self.canvas_scene.deleteCropBox()

        if self.currentActiveLayer:
            if (index == 0):
                # Move item mode is activated
                mw.status_bar.showMessage("Moving mode...", 3000)
                self.enableMoving()
            elif (index == 4):
                # Rotation mode is activated
                mw.status_bar.showMessage("Rotation mode...", 3000)
                self.enableRotating()
            elif (index == 1):
                # Resizing mode is activated
                mw.status_bar.showMessage("Resizing mode...", 3000)
                self.enableScaling()
            else:
                # Move, rotate or scale is not selected so all are disabled
                self.disable()

        if (index == 14):
            # The tool info is populated with the layer's current values
            if self.currentActiveLayer:
                red_value = (self.currentActiveLayer.getR() * 100) - 100
                green_value = (self.currentActiveLayer.getG() * 100) - 100
                blue_value = (self.currentActiveLayer.getB() * 100) - 100
            else:
                red_value, green_value, blue_value = 0, 0, 0
            self.updateRGBLabels(red_value, green_value, blue_value)
            self.updateRGBSliders(red_value, green_value, blue_value)

    def activeChanged(self):
        self.currentActiveLayer = LayerManager.getActiveLayer()
        LayerManager.disableAll()
        self.checkStackOption()

    def enableMoving(self):
        self.currentActiveLayer = LayerManager.getActiveLayer()
        self.currentActiveLayer.enableDrag()

    def enableRotating(self):
        self.currentActiveLayer = LayerManager.getActiveLayer()
        self.currentActiveLayer.enableRotate()

    def enableScaling(self):
        self.currentActiveLayer = LayerManager.getActiveLayer()
        self.currentActiveLayer.enableScale()

    def disable(self):
        self.currentActiveLayer = LayerManager.getActiveLayer()
        self.currentActiveLayer.disableAll()

    def handleAddLayerButton(self):
        # Receive the chosen filename from the user
        filepath = self.openFileNameDialog()
        if (filepath == None):
            # End process if the user has not selected a file
            return

        new_layer_name = "Layer #" + str(LayerManager.num_layers + 1)
        new_layer_z = LayerManager.num_layers

        new_image = Image.open(filepath)
        # Generate a random filename for the image
        uuid_hex = uuid.uuid4().hex
        new_file_name = uuid_hex + '.png'
        project_path.mkdir(parents=True, exist_ok=True)
        new_filepath = project_path / new_file_name
        new_image.save(new_filepath)

        # Create and add the new layer
        new_layer = LayerManager.createNewLayer(
            str(new_file_name), new_layer_name, new_layer_z, 0, 0)
        self.addLayer(new_layer)
        mw.status_bar.showMessage("New layer added...", 4000)

    def addLayer(self, layer):
        # Add layer to the canvas
        self.addLayerToCanvas(layer)
        ActionManager.layerAdded(layer)
        # Add the layer's widgets to the interface
        self.addLayerWidget(layer)
        self.addRandomiseWidget(layer)

    def checkIfLayerExists(self, file_name):
        for layer in LayerManager.layers_container:
            if file_name == layer.getImageName():
                return True
        return False

    def openFileNameDialog(self):
        options = qtw.QFileDialog.Options()
        options |= qtw.QFileDialog.DontUseNativeDialog
        file_name, _ = qtw.QFileDialog.getOpenFileName(
            self, "Digital Collage Creator - Add Image", "", "Images (*.png *.xpm *.jpg)", options=options)
        if file_name:
            return file_name

    def openProjectDialog(self):
        options = qtw.QFileDialog.Options()
        options |= qtw.QFileDialog.DontUseNativeDialog
        file_name, _ = qtw.QFileDialog.getOpenFileName(
            self, "Digital Collage Creator - Open Project", "", "Project (*.dcc)", options=options)
        if file_name:
            return file_name

    def quitApp(self):
        # Quit the app and close the window
        # Ask the user to confirm the action
        reply = qtw.QMessageBox.question(self, 'Message',
                                         "Are you sure to quit?", qtw.QMessageBox.Yes |
                                         qtw.QMessageBox.No, qtw.QMessageBox.No)

        if reply == qtw.QMessageBox.Yes:
            qtw.qApp.quit
            self.close()
        else:
            pass

    def homeInfo(self):
        self.param_section.setCurrentWidget(self.info_widget)

    def randomInfo(self):
        self.param_section.setCurrentWidget(self.randomise_container)

    def addLayerToCanvas(self, layer):
        # Add the layer image item to the canvas
        # Add the layer's rotate and scale icons to the canvas
        self.canvas_scene.addItem(layer.getLayerItem())
        layer.getLayerItem().centreItem()
        setOriginToCenter(layer.getLayerItem())
        self.addLayerIconsToCanvas(layer)

    def addLayerIconsToCanvas(self, layer):
        self.canvas_scene.addItem(layer.getRotateItem())
        self.canvas_scene.addItem(layer.getScaleItem())

    def saveCanvas(self):
        self.param_section.setCurrentWidget(self.save_widget)

    def saveSubmit(self):
        # Prompts the user for a filename input
        options = qtw.QFileDialog.Options()
        options |= qtw.QFileDialog.DontUseNativeDialog
        file_name, _ = qtw.QFileDialog.getSaveFileName(
            self, "Digital Collage Creator - Save Image", "", "Images (*.png)", options=options)

        if file_name:
            # User has entered a filename
            file_name = os.path.basename(file_name) + ".png"
            # The canvas image is drawn into a pixmap and saved as the entered filename
            pixmap_to_save = qtg.QPixmap(Canvas.width(), Canvas.height())
            painter = qtg.QPainter(pixmap_to_save)
            rect = qtc.QRect(0, 0, Canvas.width(), Canvas.height())
            self.canvas_scene.render(painter)
            painter.end()
            pixmap_to_save.save(file_name)
        mw.status_bar.showMessage("Image saved...", 4000)

    def saveProjectSubmit(self):
        # Prompts the user for a filename input
        options = qtw.QFileDialog.Options()
        options |= qtw.QFileDialog.DontUseNativeDialog
        file_name, _ = qtw.QFileDialog.getSaveFileName(
            self, "Digital Collage Creator - Save Project", "", "Text (*.dcc)", options=options)
        if file_name:
            # User has entered a filename
            # Initiate a dictionary to store the project details
            project_data = {}
            project_data['layers'] = []

            for layer in LayerManager.layers_container:
                # For each layer in the canvas add an entry to the
                # project dictionary containing layer info.

                # Convert the image into a string reperesentation
                # using base 64 encoding.
                buffered = BytesIO()
                layer_image = Image.open(
                    str(project_path / layer.getImageName()))
                layer_image.save(buffered, format="PNG")
                binary_image_data = base64.b64encode(buffered.getvalue())
                str_image_data = binary_image_data.decode('utf-8')

                # Insert the information into the dictionary
                project_data['layers'].append({
                    'x': layer.getXPosition(),
                    'y': layer.getYPosition(),
                    'z': layer.getZPosition(),
                    'image name': layer.getImageName(),
                    'layer name': layer.getLayerName(),
                    'r': layer.getR(),
                    'g': layer.getG(),
                    'b': layer.getB(),
                    'bw': layer.getBW(),
                    'blur': layer.getBlur(),
                    'sharpness': layer.getSharpness(),
                    'brightness': layer.getBrightness(),
                    'contrast': layer.getContrast(),
                    'visible': layer.getLayerWidget().is_layer_visible,
                    'rotation': layer.getLayerItem().getRotation(),
                    'scale': layer.getLayerItem().getScale(),
                    'img str': str_image_data
                })

            # Convert the dictionary content into .json format
            # Save the .json data in the filename netered by the user
            with open(file_name, 'w') as outfile:
                json.dump(project_data, outfile)
            mw.status_bar.showMessage("Project saved...", 4000)

    def openProjectSubmit(self):
        # Prompt the user for the file location of the project
        filepath = self.openProjectDialog()
        if (filepath == None):
            # User has not entered a file
            return
        if (Path(filepath).suffix != '.dcc'):
            # Ensures the filename is of the right file format
            mw.status_bar.showMessage("Unsupported file type...", 4000)
            return

        with open(filepath) as json_file:
            # Load the project json data into an dictionary
            data = json.load(json_file)
            # Loop through each layer in the data
            for layer in data['layers']:
                layer_x = layer['x']
                layer_y = layer['y']
                layer_z = layer['z']
                r = layer['r']
                g = layer['g']
                b = layer['b']
                bw = layer['bw']
                blur = layer['blur']
                sharpness = layer['sharpness']
                brightness = layer['brightness']
                contrast = layer['contrast']
                visible = layer['visible']
                rotation = layer['rotation']
                scale = layer['scale']
                layer_name = layer['layer name']
                str_image_data = layer['img str']

                # Build the image from the string representation
                t_data = str_image_data.encode('utf-8')
                img = base64.b64decode(t_data)
                image = Image.open(BytesIO(img))

                # Generate a new filename
                uuid_hex = uuid.uuid4().hex
                new_file_name = uuid_hex + '.png'
                project_path.mkdir(parents=True, exist_ok=True)
                new_filepath = project_path / new_file_name
                # Save the layer image
                image.save(new_filepath)

                # Create a new layer
                new_layer = LayerManager.createNewLayer(
                    str(new_file_name), layer_name, layer_z, layer_x, layer_y)
                self.addLayer(new_layer)

                # Set the layer properties
                new_layer.setRGB(r, g, b)
                new_layer.setBW(bw)
                new_layer.setBlur(blur)
                new_layer.setSharpness(sharpness)
                new_layer.setBrightness(brightness)
                new_layer.setContrast(contrast)
                new_layer.applyAlterations()
                new_layer.setXY(layer_x, layer_y)
                if not visible:
                    new_layer.getLayerWidget().toggleLayerVisible()
                new_layer.getLayerItem().setRotation(rotation)
                new_layer.getLayerItem().setScale(scale)

    def cutoutSubmit(self):
        # Open a cutout window for the active layer
        if LayerManager.getActiveLayer():
            image_layer = LayerManager.getActiveLayer()
            cw = CutoutWindow(image_layer)

    def gradientSubmit(self):
        number_of_colours = 7
        colour_label_1 = qtw.QLabel("<b>Colour 1:</b>")
        colour_label_2 = qtw.QLabel("<b>Colour 2:</b>")
        self.gradient_grid.addWidget(colour_label_1, 0, 0)
        self.gradient_grid.addWidget(colour_label_2, 1, 0)

        if LayerManager.getActiveLayer():
            # Get the layer's canvas image
            image_layer = LayerManager.getActiveLayer()
            altered_image_name = image_layer.getDisplayImage()

            # Get the layer's colour palette
            layer_colours = get_colours(get_image(str(project_path / altered_image_name)),
                                        number_of_colours, False)

            counter = 0
            for colour in layer_colours:
                # Convert the colour from hex representation to RGB
                rgb_colour = convertHEXtoRGB(colour)

                # Calculate the luminence value of each colour
                # Can be used to disregard colours that are too bright/dark
                #luma = (0.2126*rgb_colour[0] + 0.7152*rgb_colour[1] + 0.0722*rgb_colour[2])

                # Insert the colour into the colour choice rows 1 and 2
                colour_label1 = ColourLabel(str(rgb_colour), 1)
                colour_label1.setAlignment(qtc.Qt.AlignCenter)
                colour_label1.setFixedSize(15, 15)
                colour_label2 = ColourLabel(str(rgb_colour), 2)
                colour_label2.setAlignment(qtc.Qt.AlignCenter)
                colour_label2.setFixedSize(15, 15)
                colour_label1.updateStyleSheet()
                colour_label2.updateStyleSheet()

                self.gradient_grid.addWidget(colour_label1, 0, counter+1)
                self.gradient_grid.addWidget(colour_label2, 1, counter+1)

                counter += 1
        else:
            # Populate the colour choices as a greyscale sequence
            for i in range(0, number_of_colours):
                col = 255 / (i+1)
                col = str(col)
                col = "({},{},{})".format(col, col, col)
                colour_label1 = ColourLabel(col, 1)
                colour_label1.setAlignment(qtc.Qt.AlignCenter)
                colour_label1.setFixedSize(15, 15)
                colour_label2 = ColourLabel(col, 2)
                colour_label2.setAlignment(qtc.Qt.AlignCenter)
                colour_label2.setFixedSize(15, 15)
                colour_label1.updateStyleSheet()
                colour_label2.updateStyleSheet()
                self.gradient_grid.addWidget(colour_label1, 0, i+1)
                self.gradient_grid.addWidget(colour_label2, 1, i+1)

    def gradientGenerate(self):
        # Get the colour selections
        c1 = GradientManager.getActiveColour(1)
        c2 = GradientManager.getActiveColour(2)
        # Convert the string RGB representation into a [r,g,b] tuple
        c1 = convertRGBStrToTuple(c1)
        c2 = convertRGBStrToTuple(c2)

        # Get an array containing the colour values of the gradient image
        gradient_colours = gradient_array(
            Canvas.width(), Canvas.height(), c1, c2, (False, False, False))

        # Create and add a layer containing the new gradient image
        layer_number = LayerManager.num_layers + 1
        file_name = 'gradient' + str(layer_number) + '.png'
        Image.fromarray(np.uint8(gradient_colours)).save(
            str(project_path / file_name))

        new_layer_name = "Layer #" + str(LayerManager.num_layers + 1)
        new_layer_z = LayerManager.num_layers
        new_layer = LayerManager.createNewLayer(
            file_name, new_layer_name, new_layer_z, 0, 0)
        self.addLayer(new_layer)

    def undoAction(self):
        ActionManager.undoClick()

    def redoAction(self):
        ActionManager.redoClick()

    def handleCutoutButton(self):
        self.param_section.setCurrentWidget(self.cutout_widget)

    def handleGradientButton(self):
        self.param_section.setCurrentWidget(self.gradient_widget)

    def handleBrightnessButton(self):
        self.param_section.setCurrentWidget(self.brightness_widget)

    def handleAlignLayerButton(self):
        self.param_section.setCurrentWidget(self.alignment_widget)

    def handleAlignVTop(self):
        if LayerManager.getActiveLayer():
            LayerManager.getActiveLayer().alignVTop()
            mw.status_bar.showMessage("Layer aligned...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleAlignVCenter(self):
        if LayerManager.getActiveLayer():
            LayerManager.getActiveLayer().alignVCenter()
            mw.status_bar.showMessage("Layer aligned...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleAlignVBottom(self):
        if LayerManager.getActiveLayer():
            LayerManager.getActiveLayer().alignVBottom()
            mw.status_bar.showMessage("Layer aligned...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleAlignHLeft(self):
        if LayerManager.getActiveLayer():
            LayerManager.getActiveLayer().alignHLeft()
            mw.status_bar.showMessage("Layer aligned...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleAlignHCenter(self):
        if LayerManager.getActiveLayer():
            LayerManager.getActiveLayer().alignHCenter()
            mw.status_bar.showMessage("Layer aligned...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleAlignHRight(self):
        if LayerManager.getActiveLayer():
            LayerManager.getActiveLayer().alignHRight()
            mw.status_bar.showMessage("Layer aligned...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def brightnessSubmit(self):
        factor = self.brightness_param_factor.value()
        factor = factor/100
        factor = 1 + factor
        if (LayerManager.getActiveLayer()):
            orig_brightness = LayerManager.getActiveLayer().brightness
            new_brightness = float(factor)
            ActionManager.brightnessChanged(
                LayerManager.getActiveLayer(), orig_brightness, new_brightness)
            LayerManager.getActiveLayer().setBrightness(new_brightness)
            LayerManager.getActiveLayer().applyAlterations()
            mw.status_bar.showMessage("Brightness changed...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def rgbSubmit(self):
        if (LayerManager.getActiveLayer()):
            r = self.rgb_r_factor.value()
            r = r/100
            r = 1 + r
            g = self.rgb_g_factor.value()
            g = g/100
            g = 1 + g
            b = self.rgb_b_factor.value()
            b = b/100
            b = 1 + b
            active_layer = LayerManager.getActiveLayer()
            layer_rgb = active_layer.rgb
            orig_rgb = [layer_rgb[0], layer_rgb[1], layer_rgb[2]]
            new_rgb = [r, g, b]
            ActionManager.rgbChanged(active_layer, orig_rgb, new_rgb)
            LayerManager.getActiveLayer().setRGB(r, g, b)
            LayerManager.getActiveLayer().applyAlterations()
            mw.status_bar.showMessage("Image colour changed...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleBawButton(self):
        self.param_section.setCurrentWidget(self.baw_widget)

    def submitBaw(self):
        if (LayerManager.getActiveLayer()):
            orig_baw = LayerManager.getActiveLayer().bw
            if orig_baw:
                new_baw = False
                mw.status_bar.showMessage(
                    "Black & White filter removed...", 3000)
            else:
                new_baw = True
                mw.status_bar.showMessage(
                    "Black & White filter applied...", 3000)
            ActionManager.bawChanged(
                LayerManager.getActiveLayer(), orig_baw, new_baw)
            LayerManager.getActiveLayer().setBW(new_baw)
            LayerManager.getActiveLayer().applyAlterations()
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleCropButton(self):
        self.param_section.setCurrentWidget(self.crop_widget)

    def cropSubmit(self):
        if (LayerManager.getActiveLayer() and self.canvas_scene.getCurrentCropBox()):
            self.canvas_scene.deleteCropBox()
            layer = LayerManager.getActiveLayer()
            layer_item = LayerManager.getActiveLayer().getLayerItem().sceneBoundingRect()
            crop_box = self.canvas_scene.getCurrentCropBox().sceneBoundingRect()

            # If the user starts drawing the cropbox from the bottom
            # or the left the cropbox will have a negative width/height.
            # A bounding box (x0, y0, x1, y1) is created using the
            # crop box width, height and starting coordinates.
            if (crop_box.width() < 0):
                x1 = crop_box.x() + crop_box.width()
                x2 = crop_box.x()
            else:
                x1 = crop_box.x()
                x2 = crop_box.x() + crop_box.width()

            if (crop_box.height() < 0):
                y1 = crop_box.y() + crop_box.height()
                y2 = crop_box.y()
            else:
                y1 = crop_box.y()
                y2 = crop_box.y() + crop_box.height()

            # Limits the crop box to the boundaries of the layer image
            if (x1 < layer_item.x()):
                x1 = layer_item.x()
            if (x2 > (layer_item.x() + layer_item.width())):
                x2 = layer_item.x() + layer_item.width()
            if (y1 < layer_item.y()):
                y1 = layer_item.y()
            if (y2 > (layer_item.y() + layer_item.height())):
                y2 = layer_item.y() + layer_item.height()

            originPoint = qtc.QPointF(x1, y1)

            x1 -= layer_item.x()
            x2 -= layer_item.x()
            y1 -= layer_item.y()
            y2 -= layer_item.y()

            layer.crop(x1, y1, x2, y2, originPoint)
            mw.status_bar.showMessage("Layer cropped...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleDeleteButton(self):
        self.param_section.setCurrentWidget(self.delete_widget)

    def deleteSubmit(self):
        if (LayerManager.getActiveLayer()):
            # If there is an active layer the LayerManager is called to delete it
            ActionManager.layerDeleted(LayerManager.getActiveLayer())
            self.deleteLayer(LayerManager.getActiveLayer())
            mw.status_bar.showMessage("Layer deleted...", 4000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def deleteLayer(self, layer):
        # Remove the layer from the canvas
        LayerManager.tempDeleteLayer(layer)
        # Move the layers above the deleted layer down one
        deleted_z_index = layer.getZPosition()
        if layer in LayerManager.layers_container:
            layer.disableVisible()
            LayerManager.layers_container.remove(layer)
            LayerManager.num_layers -= 1
        for layer in LayerManager.layers_container:
            if layer.getZPosition() > deleted_z_index:
                layer.setZPosition(layer.getZPosition() - 1)

    def handleMoveLayerButton(self):
        self.param_section.setCurrentWidget(self.move_param_widget)

    def handleRotateLayerButton(self):
        self.param_section.setCurrentWidget(self.rotate_widget)

    def handleResizeLayerButton(self):
        self.param_section.setCurrentWidget(self.resize_factor_widget)

    def handleRGBEditButton(self):
        self.param_section.setCurrentWidget(self.rgb_widget)

    def handleBlurButton(self):
        self.param_section.setCurrentWidget(self.blur_widget)

    def blurSubmit(self):
        if (LayerManager.getActiveLayer()):
            # Activate/remove the blur effect on the active layer
            orig_blur = LayerManager.getActiveLayer().blur
            if orig_blur:
                new_blur = False
                mw.status_bar.showMessage("Blur filter removed...", 3000)
            else:
                new_blur = True
                mw.status_bar.showMessage("Blur filter applied...", 3000)
            ActionManager.blurChanged(
                LayerManager.getActiveLayer(), orig_blur, new_blur)
            LayerManager.getActiveLayer().setBlur(new_blur)
            LayerManager.getActiveLayer().applyAlterations()
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleContrastButton(self):
        self.param_section.setCurrentWidget(self.contrast_widget)

    def contrastSubmit(self):
        factor = self.contrast_param_factor.value()
        factor = factor/100
        factor = 1 + factor
        if (LayerManager.getActiveLayer()):
            orig_contrast = LayerManager.getActiveLayer().contrast
            new_contrast = factor
            ActionManager.contrastChanged(
                LayerManager.getActiveLayer(), orig_contrast, new_contrast)
            LayerManager.getActiveLayer().setContrast(new_contrast)
            LayerManager.getActiveLayer().applyAlterations()
            mw.status_bar.showMessage("Contrast changed...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleDetailsButton(self):
        self.param_section.setCurrentWidget(self.details_widget)

    def detailsSubmit(self):
        factor = self.details_param_factor.value()
        factor = factor/100
        factor = 1 + factor
        if (LayerManager.getActiveLayer()):
            orig_sharpness = LayerManager.getActiveLayer().sharpness
            new_sharpness = factor
            ActionManager.sharpnessChanged(
                LayerManager.getActiveLayer(), orig_sharpness, new_sharpness)
            LayerManager.getActiveLayer().setSharpness(new_sharpness)
            LayerManager.getActiveLayer().applyAlterations()
            mw.status_bar.showMessage("Sharpness changed...", 3000)
        else:
            mw.status_bar.showMessage(
                "No active layer selected, activate a layer from the Layer Manager...", 4000)

    def handleAddTextButton(self):
        self.param_section.setCurrentWidget(self.text_widget)

    def textSubmit(self):
        text = self.text_param_text.text()
        font = self.text_param_font.text()
        colour = "#" + self.text_param_colour.text()
        rgb_colour = ImageColor.getcolor(colour, "RGB")
        size = self.text_param_size.text()
        if text != "":
            # Create and add a new layer containing the text
            new_text_layer = createText(font, float(size), text, rgb_colour)
            self.addLayer(new_text_layer)
            mw.status_bar.showMessage("Text added to canvas...", 4000)
        else:
            mw.status_bar.showMessage("You must enter some text...", 4000)

    def handleShapesButton(self):
        return

    def handleFiltersButton(self):
        self.param_section.setCurrentWidget(self.filter_widget)

    def handleColoursButton(self):
        self.param_section.setCurrentWidget(self.colours_widget)

    def coloursSubmit(self):
        factor = self.colours_param_factor.text()
        factor = float(factor)
        if (LayerManager.getActiveLayer()):
            enhanceColour(LayerManager.getActiveLayer(), factor)
            LayerManager.getActiveLayer().getLayerWidget().updateThumbnail()

    def updateCanvas(self, new_image_name):
        self.image = qtg.QPixmap(new_image_name)
        self.image = self.image.scaled(self.image_label.frameGeometry().width(
        ), self.image_label.frameGeometry().height(), qtc.Qt.KeepAspectRatio, qtc.Qt.FastTransformation)
        self.image_label.setPixmap(self.image)


def alterRGB(image_name, r, g, b):
    # Open the image in the provided file location
    image = Image.open(str(project_path / image_name))
    image = image.convert("RGBA")

    # Get the red, green and blue channel bands of the image
    source = image.split()
    R, G, B = 0, 1, 2

    # Multiply the channel by the provided factor
    outR = source[R].point(lambda i: i * r)
    outG = source[G].point(lambda i: i * g)
    outB = source[B].point(lambda i: i * b)

    # Replace the original image channels
    source[R].paste(outR)
    source[G].paste(outG)
    source[B].paste(outB)

    # Merge the channels together and replace the original image
    rgb_image = Image.merge("RGB", (source[R], source[G], source[B]))
    image.paste(rgb_image, mask=image)
    return image


def createText(fontName, size, text, colour):
    font = ImageFont.truetype(fontName, int(size))
    # Create a blank image of size 0
    text_image = Image.new("RGBA", (0, 0), color=(0, 0, 0, 0))

    # Calculate the size of the text
    draw = ImageDraw.Draw(text_image)
    w, h = draw.textsize(text=text, font=font)

    # Resize the blank image to contain the text
    text_image = text_image.resize((w, h))
    text_image.paste((0, 0, 0, 0), [0, 0, w, h])

    # Draw the text into the image
    draw = ImageDraw.Draw(text_image)
    draw.text((0, 0), text, font=font, fill=colour)

    # Save the image
    new_layer_name = "Layer #" + str(LayerManager.num_layers + 1)
    new_image_name = "text_" + str(LayerManager.num_layers + 1) + ".png"
    new_layer_z = LayerManager.num_layers
    text_image.save(str(project_path / new_image_name))
    # Create and return a layer containing the text image
    new_text_layer = LayerManager.createNewLayer(
        new_image_name, new_layer_name, new_layer_z, 0, 0)
    return new_text_layer


def cropImage(image, startX, startY, endX, endY):
    # Returns the region of the image specified by
    # the bounding box.
    box = (startX, startY, endX, endY)
    cropped = image.copy()
    region = cropped.crop(box)
    return region


def blurImage(image_name):
    # Apply the blur filter to the specified image file
    image = Image.open(str(project_path / image_name))
    blur = image.copy().filter(ImageFilter.BLUR)
    return blur


def enhanceContrast(image_name, factor):
    # Enhance the contrast of the provided image
    # using the provided factor.
    image = Image.open(str(project_path / image_name))
    image_enh = ImageEnhance.Contrast(image.copy())
    enhanced_image = image_enh.enhance(factor)
    return enhanced_image


def enhanceBrightness(image_name, factor):
    # Enhance the brightness of the provided image
    # using the provided factor.
    image = Image.open(str(project_path / image_name))
    image_enh = ImageEnhance.Brightness(image.copy())
    enhanced_image = image_enh.enhance(factor)
    return enhanced_image


def enhanceColour(image_name, factor):
    # Enhance the colour of the provided image
    # using the provided factor.
    image = Image.open(str(project_path / image_name))
    image_enh = ImageEnhance.Color(image.copy())
    enhanced_image = image_enh.enhance(factor)
    return enhanced_image


def enhanceSharpness(image_name, factor):
    # Enhance the sharpness of the provided image
    # using the provided factor.
    image = Image.open(str(project_path / image_name))
    image_enh = ImageEnhance.Sharpness(image.copy())
    enhanced_image = image_enh.enhance(factor)
    return enhanced_image


def makeLayerBaW(image_name):
    # Apply a black and white filter to the provided image
    image = Image.open(str(project_path / image_name))
    image_alpha = image.split()[-1]
    grey = image.copy().convert("L")
    grey.convert("RGB")
    grey.putalpha(image_alpha)
    return grey


def convertRGBtoHEX(color):
    # Convert an RGB array into a Hex string value
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def convertHEXtoRGB(h):
    # Convert a hex string value into an RGB array
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))


def convertRGBStrToTuple(rgb):
    # Convert an RGB string into an RGB array
    rgb = rgb.replace("(", "")
    rgb = rgb.replace(")", "")
    rgb = tuple(map(float, rgb.split(",")))
    rgb = tuple(map(int, rgb))
    return rgb


def get_image(image_path):
    # Get an image using the cv2 library
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_colours(image, number_of_colors, show_chart):
    # Resize the image
    resized_image = cv2.resize(
        image, (500, 500), interpolation=cv2.INTER_AREA)
    resized_image = resized_image.reshape(
        resized_image.shape[0]*resized_image.shape[1], 3)

    # Use K-Means clustering to get the most dominant colours
    colours_kmeans = KMeans(n_clusters=number_of_colors)
    colour_labels = colours_kmeans.fit_predict(resized_image)
    counts = Counter(colour_labels)

    center_colours = colours_kmeans.cluster_centers_
    ordered_colours = [center_colours[i] for i in counts.keys()]
    hex_colours = [convertRGBtoHEX(ordered_colours[i])
                   for i in range(0, len(ordered_colours))]

    return hex_colours


def gradient_array_channel(start, stop, width, height, horizontal_bool):
    if horizontal_bool:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_array(width, height, start_list, stop_list, horizontal_bool):
    # Create an array populate with zeros
    gradient_array = np.zeros((height, width, len(start_list)), dtype=np.float64)

    for i, (start, stop, horizontal_bool) in enumerate(zip(start_list, stop_list, horizontal_bool)):
        gradient_array[:, :, i] = gradient_array_channel(
            start, stop, width, height, horizontal_bool)

    return gradient_array


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)

    # Create a project directory
    shutil.rmtree(project_path, ignore_errors=True)

    # Display a splash loading screen
    splash_screen_image = qtg.QPixmap(':/icon_logo.png')
    splash_screen = qtw.QSplashScreen(
        splash_screen_image, qtc.Qt.WindowStaysOnTopHint)
    splash_screen.setWindowFlags(
        qtc.Qt.WindowStaysOnTopHint | qtc.Qt.FramelessWindowHint)
    splash_screen.setEnabled(False)
    splash_screen.show()

    app.processEvents()
    mw = MainWindow()
    splash_screen.finish(mw)
    sys.exit(app.exec())
