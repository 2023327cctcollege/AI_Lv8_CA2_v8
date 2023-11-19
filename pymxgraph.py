"""
AI_Lv8_CA2_v8

CCT College Dublin  
Bachelor of Science Honours in Computing in Information Technology  
Introduction to Artificial Intelligence - Y4M1  
Year 4, Semester 7  
Continuous Assessment 2

Lecturer name: David McQuaid  
Lecturer email: dmcquaid@cct.ie

Student Name: Mateus Fonseca Campos  
Student Number: 2023327  
Student Email: 2023327@student.cct.ie

Submission date: 19 November 2023

GitHub: https://github.com/2023327cctcollege/AI_Lv8_CA2_v8
"""

import copy
import math
import re
import xml.etree.ElementTree as ET

import cairo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import deque
from PIL import Image
from PIL import ImageColor
from IPython.display import HTML


class MGX:
    def __init__(self, file=None):
        self.graph = {}
        self.graph_backup = {}
        self.vertices = []
        self.edges = []
        self.weights = []
        self.adj_list = {}
        self.coordinates = {}
        self.px = 0
        self.py = 0
        self.dx = 0
        self.dy = 0
        self.plot = None
        self.frames = [[]]

        if file:
            self.parse(file)

    def parse(self, file):
        tree = ET.parse(file)

        parse = lambda x: {
            k: (float(v) if v.replace(".", "").isnumeric() else v) for k, v in x.items()
        }

        for el in list(tree.iter())[2:]:
            curr_k = el.get("id", el.tag)

            if el.tag == "mxGeometry":
                self.graph[prev_k][curr_k] = parse(el)
            else:
                self.graph[curr_k] = parse(el)

            prev_k = curr_k

            if el.get("edge"):
                self.edges.append(el.get("id"))
            elif el.get("vertex"):
                if not el.get("parent").isnumeric():
                    self.weights.append(el.get("id"))
                elif el.get("style") != "group":
                    self.vertices.append(el.get("id"))

        self.graph_backup = copy.deepcopy(self.graph)

        self.px, self.dx = (lambda x: (int(min(x)), int(max(x))))(
            [self.graph[vertex]["mxGeometry"]["x"] for vertex in self.vertices]
        )
        self.py, self.dy = (lambda x: (int(min(x)), int(max(x))))(
            [self.graph[vertex]["mxGeometry"]["y"] for vertex in self.vertices]
        )
        self.dx += int(self.px + self.graph[self.vertices[0]]["mxGeometry"]["width"])
        self.dy += int(self.py + self.graph[self.vertices[0]]["mxGeometry"]["height"])

        self.get_adj_list()

        self.coordinates = {
            self.graph[vertex]["value"]: {
                "x": self.graph[vertex]["mxGeometry"]["x"],
                "y": self.graph[vertex]["mxGeometry"]["y"],
            }
            for vertex in self.vertices
        }

        self.plot = self.paint()

    def get_adj_list(self):
        aux = {vertex: self.graph[vertex]["value"] for vertex in self.vertices}
        aux2 = {
            self.graph[weight]["parent"]: self.graph[weight]["value"]
            for weight in self.weights
        }

        for edge in self.edges:
            self.adj_list[aux[self.graph[edge]["source"]]] = self.adj_list.get(
                "source", {}
            )
            self.adj_list[aux[self.graph[edge]["target"]]] = self.adj_list.get(
                "target", {}
            )

        for edge in self.edges:
            self.adj_list[aux[self.graph[edge]["source"]]][
                aux[self.graph[edge]["target"]]
            ] = int(aux2[self.graph[edge]["parent"]])
            self.adj_list[aux[self.graph[edge]["target"]]][
                aux[self.graph[edge]["source"]]
            ] = int(aux2[self.graph[edge]["parent"]])

    def render(self, flag=0, path=False, graph=None, n=0):
        if len(self.frames) - 1 < n:
            self.frames.append([self.frames[0][0]])

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.dx, self.dy)
        ctx = cairo.Context(surface)

        for edge in self.edges:
            s_geo = graph[graph[edge]["source"]]["mxGeometry"]
            t_geo = graph[graph[edge]["target"]]["mxGeometry"]

            b, g, r = ImageColor.getcolor(
                list(
                    filter(
                        lambda x: x.startswith("strokeColor"),
                        graph[edge]["style"].split(";"),
                    )
                )[0][-7:],
                "RGB",
            )

            w = list(
                map(
                    lambda x: int(x),
                    list(
                        filter(
                            lambda x: x.startswith("strokeWidth"),
                            graph[edge]["style"].split(";"),
                        )
                    )[0][-1:],
                )
            )[0]

            ctx.move_to(
                s_geo["x"] + s_geo["width"] / 2, s_geo["y"] + s_geo["height"] / 2
            )
            ctx.line_to(
                t_geo["x"] + t_geo["width"] / 2, t_geo["y"] + t_geo["height"] / 2
            )

            ctx.set_source_rgb(r / 255, g / 255, b / 255)
            ctx.set_line_width(w)
            ctx.stroke()

        for vertex in self.vertices:
            geo = graph[vertex]["mxGeometry"]

            b, g, r = ImageColor.getcolor(
                list(
                    filter(
                        lambda x: x.startswith("fillColor"),
                        graph[vertex]["style"].split(";"),
                    )
                )[0][-7:],
                "RGB",
            )

            ctx.arc(
                geo["x"] + geo["width"] / 2,
                geo["y"] + geo["height"] / 2,
                geo["width"] / 2,
                0,
                2 * math.pi,
            )
            ctx.set_source_rgb(r / 255, g / 255, b / 255)
            ctx.fill()
            ctx.stroke()

            b, g, r = ImageColor.getcolor(
                list(
                    filter(
                        lambda x: x.startswith("fontColor"),
                        graph[vertex]["style"].split(";"),
                    )
                )[0][-7:],
                "RGB",
            )

            ctx.set_source_rgb(r / 255, g / 255, b / 255)
            ctx.set_font_size(12)
            ctx.select_font_face(
                "Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
            )

            value = graph[vertex]["value"]
            _, _, width, height, _, _ = ctx.text_extents(value)

            ctx.move_to(
                geo["x"] + geo["width"] / 2 - width / 2,
                geo["y"] + geo["height"] / 2 + height / 2,
            )
            ctx.show_text(value)

        for weight in self.weights:
            b, g, r = ImageColor.getcolor(
                list(
                    filter(
                        lambda x: x.startswith("fillColor"),
                        graph[weight]["style"].split(";"),
                    )
                )[0][-7:],
                "RGB",
            )

            geo = graph[weight]["mxGeometry"]
            p_geo = graph[graph[weight]["parent"]]["mxGeometry"]

            ctx.rectangle(p_geo["x"], p_geo["y"], geo["width"], geo["height"])
            ctx.set_source_rgb(r / 255, g / 255, b / 255)
            ctx.fill()
            ctx.stroke()

            b, g, r = ImageColor.getcolor(
                list(
                    filter(
                        lambda x: x.startswith("fontColor"),
                        graph[weight]["style"].split(";"),
                    )
                )[0][-7:],
                "RGB",
            )

            ctx.set_source_rgb(r / 255, g / 255, b / 255)
            ctx.set_font_size(12)
            ctx.select_font_face(
                "Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
            )

            value = str(int(graph[weight]["value"]))
            _, _, width, height, _, _ = ctx.text_extents(value)

            ctx.move_to(
                p_geo["x"] + geo["width"] / 2 - width / 2,
                p_geo["y"] + geo["height"] / 2 + height / 2,
            )
            ctx.show_text(value)

        buf = surface.get_data()
        img = np.ndarray(shape=[self.dy, self.dx, 4], dtype=np.uint8, buffer=buf)
        self.frames[n].append(img)

        return img

    def paint(self, path=None, flag=0, every_step=True, n=1):
        if not path:
            return self.render(flag, graph=self.graph)

        try:
            path = {v: path[i - 1] if i > 0 else None for i, v in enumerate(path)}
            inv_aux = {self.graph[vertex]["value"]: vertex for vertex in self.vertices}

            for p in path:
                target = p
                source = path[p]

                if source == None or target == None:
                    continue

                try:
                    v = list(
                        filter(
                            lambda x: self.graph[x]["source"] == inv_aux[source]
                            and self.graph[x]["target"] == inv_aux[target],
                            self.edges,
                        )
                    )[0]

                except:
                    v = list(
                        filter(
                            lambda x: self.graph[x]["source"] == inv_aux[target]
                            and self.graph[x]["target"] == inv_aux[source],
                            self.edges,
                        )
                    )[0]

                if len(v) != 0:
                    self.graph[v]["style"] = re.sub(
                        r"strokeColor=[^;]*;",
                        "strokeColor=#ff0000;",
                        self.graph[v]["style"],
                    )

                    if every_step:
                        self.render(flag, path=True, graph=self.graph, n=n)

            if not every_step:
                self.render(flag, path=True, graph=self.graph, n=n)

        except:
            for index, pa in enumerate(path):
                self.graph = copy.deepcopy(self.graph_backup)
                self.paint(path=pa, n=index)

    def show(self, paths=None, animated=False, title=None, labels=None):
        fig = plt.figure()

        if paths:
            self.paint(paths, animated)
            axs = []

            for i, _ in enumerate(paths):
                axs.append(
                    plt.subplot(
                        1,
                        len(paths),
                        i + 1,
                        frameon=False,
                        xlabel=title,
                        xticks=[],
                        yticks=[],
                    )
                )

            if animated:

                def animator(fr):
                    for i, ax in enumerate(axs):
                        try:
                            ax.imshow(self.frames[i][fr])
                        except:
                            ax.imshow(self.frames[i][-1])

                ani = animation.FuncAnimation(
                    fig,
                    animator,
                    frames=max([len(path) for path in paths]),
                    interval=1000,
                    repeat_delay=1000,
                )

                return HTML(ani.to_jshtml())

            else:
                for i, ax in enumerate(axs):
                    ax.imshow(self.frames[i][-1])

                return plt.show()

        plt.imshow(self.plot)

        return plt.show()

    def dijkstra(self, source, target):
        q = deque()
        path = []
        previous = {}
        dist = {source: 0}

        for v in self.adj_list:
            if v != source:
                dist[v] = float("infinity")
            q.append(v)

        while len(q) > 0:
            v = min(q, key=lambda v: dist[v])

            if v == target:
                path.append(v)
                while previous[v] != source:
                    path.append(previous[v])
                    v = previous[v]
                path.append(previous[v])
                path.reverse()
                return path

            q.remove(v)

            for u in self.adj_list[v]:
                alt = dist[v] + self.adj_list[v][u]
                if alt < dist[u]:
                    previous[u] = v
                    dist[u] = alt

        return dist
