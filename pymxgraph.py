class MGX:
    def __init__(self):
        self.graph = {}
        self.vertices = []
        self.edges = []
        self.weights = []
        self.adj_list = {}
        self.px = 0
        self.py = 0
        self.dx = 0
        self.dy = 0
        self.frames = []

    def parse(self, path):
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(path)

        parse = lambda x: {k: (float(v) if v.replace(".", "").isnumeric() else v) for k, v in x.items()}

        for el in list(tree.iter())[2:]:
            curr_k = el.get('id', el.tag)
            
            if el.tag == 'mxGeometry':
                self.graph[prev_k][curr_k] = parse(el)
            else:
                self.graph[curr_k] = parse(el)
                
            prev_k = curr_k

            if el.get('edge'):
                self.edges.append(el.get('id'))
            elif el.get('vertex'):
                if not el.get('parent').isnumeric():
                    self.weights.append(el.get('id'))
                elif el.get('style') != 'group':
                    self.vertices.append(el.get('id'))

        self.px, self.dx = (lambda x: (int(min(x)), int(max(x))))([self.graph[vertex]['mxGeometry']['x'] for vertex in self.vertices])
        self.py, self.dy = (lambda x: (int(min(x)), int(max(x))))([self.graph[vertex]['mxGeometry']['y'] for vertex in self.vertices])
        self.dx += int(self.px + self.graph[self.vertices[0]]['mxGeometry']['width'])
        self.dy += int(self.py + self.graph[self.vertices[0]]['mxGeometry']['height'])
        
        self.get_adj_list()

    def get_adj_list(self):
        aux = {vertex: self.graph[vertex]['value'] for vertex in self.vertices}
        aux2 = {self.graph[weight]['parent']: self.graph[weight]['value'] for weight in self.weights}        
        
        for edge in self.edges:
            self.adj_list[aux[self.graph[edge]['source']]] = self.adj_list.get('source', {})
            self.adj_list[aux[self.graph[edge]['target']]] = self.adj_list.get('target', {})

        for edge in self.edges:
            self.adj_list[aux[self.graph[edge]['source']]][aux[self.graph[edge]['target']]] = int(aux2[self.graph[edge]['parent']])
            self.adj_list[aux[self.graph[edge]['target']]][aux[self.graph[edge]['source']]] = int(aux2[self.graph[edge]['parent']])
    
    def render(self, save_to, flag, path=None, graph=None):
        if not graph:
            if not path:
                raise ValueError
            else:
                graph = self.parse(path)

        if flag == 0: self.frames = []

        import cairo
        import math
        from PIL import ImageColor
        import numpy as np

        # with cairo.SVGSurface(save_to, self.dx, self.dy) as surface: 
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.dx, self.dy)
        ctx = cairo.Context(surface)

        # for edge in filter(lambda x: 'edge' in graph[x].keys() and graph[x]['edge'] == 1, graph):
        for edge in self.edges:
            s_geo = graph[graph[edge]['source']]['mxGeometry']
            t_geo = graph[graph[edge]['target']]['mxGeometry']
            
            b, g, r = ImageColor.getcolor(
                list(filter(lambda x: x.startswith('strokeColor'), graph[edge]['style'].split(';')))[0][-7:], 
                "RGB"
            )
            
            w = list(map(lambda x: int(x), list(filter(lambda x: x.startswith('strokeWidth'), graph[edge]['style'].split(';')))[0][-1:]))[0]

            ctx.move_to(s_geo['x'] + s_geo['width']/2, s_geo['y'] + s_geo['height']/2)
            ctx.line_to(t_geo['x'] + t_geo['width']/2, t_geo['y'] + t_geo['height']/2)

            ctx.set_source_rgb(r/255, g/255, b/255)
            ctx.set_line_width(w)
            ctx.stroke()
        
        # for vertex in filter(lambda x: 'vertex' in graph[x].keys() and graph[x]['vertex'] == 1 and graph[x]['style'] != 'group', graph):
        for vertex in self.vertices:
            geo = graph[vertex]['mxGeometry']

            # if graph[vertex]['parent'] == 1:
            b, g, r = ImageColor.getcolor(
                list(filter(lambda x: x.startswith('fillColor'), graph[vertex]['style'].split(';')))[0][-7:], 
                "RGB"
            )

            ctx.arc(
                geo['x'] + geo['width']/2,
                geo['y'] + geo['height']/2,
                geo['width']/2,
                0,
                2*math.pi
            )
            ctx.set_source_rgb(r/255, g/255, b/255)
            ctx.fill()
            ctx.stroke()

            ######################

            b, g, r = ImageColor.getcolor(
                list(filter(lambda x: x.startswith('fontColor'), graph[vertex]['style'].split(';')))[0][-7:], 
                "RGB"
            )

            ctx.set_source_rgb(r/255, g/255, b/255)
            ctx.set_font_size(12)
            ctx.select_font_face("Helvetica",
                                cairo.FONT_SLANT_NORMAL,
                                cairo.FONT_WEIGHT_BOLD)

            value = graph[vertex]['value']
            _, _, width, height, _, _ = ctx.text_extents(value)

            ctx.move_to(
                geo['x'] + geo['width']/2 - width/2,
                geo['y'] + geo['height']/2 + height/2
            )
            ctx.show_text(value)
            # else:

        for weight in self.weights:
            b, g, r = ImageColor.getcolor(
                list(filter(lambda x: x.startswith('fillColor'), graph[weight]['style'].split(';')))[0][-7:], 
                "RGB"
            )
                            
            geo = graph[weight]['mxGeometry']
            p_geo = graph[graph[weight]['parent']]['mxGeometry']

            ctx.rectangle(p_geo['x'], p_geo['y'], geo['width'], geo['height'])
            ctx.set_source_rgb(r/255, g/255, b/255)
            ctx.fill()
            ctx.stroke()

            ######################

            b, g, r = ImageColor.getcolor(
                list(filter(lambda x: x.startswith('fontColor'), graph[weight]['style'].split(';')))[0][-7:], 
                "RGB"
            )

            ctx.set_source_rgb(r/255, g/255, b/255)
            ctx.set_font_size(12)
            ctx.select_font_face("Helvetica",
                                cairo.FONT_SLANT_NORMAL,
                                cairo.FONT_WEIGHT_BOLD)

            value = str(int(graph[weight]['value']))
            _, _, width, height, _, _ = ctx.text_extents(value)

            ctx.move_to(
                p_geo['x'] + geo['width']/2 - width/2,
                p_geo['y'] + geo['height']/2 + height/2
            )
            ctx.show_text(value)

        buf = surface.get_data()
        
        self.frames.append(np.ndarray(shape=[self.dy, self.dx, 4], dtype=np.uint8, buffer=buf))

        # im = np.frombuffer(buf, dtype=np.uint8)
        # im = np.reshape(im, [self.dy, self.dx, 4])

        # self.frames.append(im)

    def show(path):
        from IPython.display import SVG, display

        display(SVG(path))
