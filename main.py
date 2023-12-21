import neural
import pygame
import numpy
import math
import imgui
import imgui.integrations.pygame



def float_color(rgb):
    return [ch / 255 for ch in rgb] + [1]



class Trace:

    def __init__(self, points=[]):
        self.points = points[:]

    def add_point(self, point):
        point = numpy.array(point)
        if len(self.points) == 0 or any(point != self.points[-1]):
            self.points.append(point)

    def vectorize(self, size):
        # collapse nearest points
        new_points = self.points[1:-1]
        while len(new_points) > size // 2 - 1:
            distances = [math.hypot(*numpy.subtract(*new_points[i:i + 2])) for i in range(len(new_points) - 1)]
            closest_index = distances.index(min(distances))
            points = new_points[closest_index:closest_index + 2]
            center = sum(points) / 2
            new_points.pop(closest_index)
            new_points[closest_index] = center
        new_points = [self.points[0]] + new_points + [self.points[-1]]

        # create normalized vectors
        vectors = [-numpy.subtract(*new_points[i:i + 2]) for i in range(len(new_points) - 1)]
        normalized = [vector / numpy.linalg.norm(vector) for vector in vectors]

        # return column vector
        vector_array = numpy.array(normalized)
        return vector_array.flatten().reshape(-1, 1)

    def from_vector(vector, trace_scale, trace_offset):
        # trace vectors
        vectors = [*vector.reshape((len(vector) // 2, 2))]
        points = [numpy.zeros(2)]
        for vector in vectors:
            points.append(points[-1] + vector)

        # scale points
        point_array = numpy.array(points)
        top_left = numpy.amin(point_array, 0)
        bottom_right = numpy.amax(point_array, 0)
        size = bottom_right - top_left
        scale = numpy.max(size)
        center = top_left + size / 2
        offset = center - scale / 2
        normalized = (point_array - offset) / scale
        scaled = [*(normalized * trace_scale + trace_offset)]
        
        return Trace(scaled)

    def is_valid(self):
        return len(self.points) >= 2

    def draw(self, draw_list, line_color, line_width):
        color = imgui.get_color_u32_rgba(*float_color(line_color))
        lines = [self.points[i:i + 2] for i in range(len(self.points) - 1)]
        for line in lines:
            draw_list.add_line(*line[0], *line[1], color, line_width)
            draw_list.add_circle_filled(*line[0], line_width / 2, color)
        draw_list.add_circle_filled(*lines[-1][1], line_width / 2, color)



class App:

    CANVAS_MODE_TRAINING = 0
    CANVAS_MODE_INFERENCE = 1

    WINDOW_FPS = 60
    WINDOW_SIZE = (600, 600)
    LINE_COLOR = (0, 0, 0)
    LINE_WIDTH = 6
    BACKGROUND_COLOR_TRAINING = (255, 255, 200)
    BACKGROUND_COLOR_INFERENCE = (255, 255, 255)

    LEARNING_EPOCHS_INIT = 1000
    LEARNING_RATE_INIT = 0.05
    ARCHITECTURE_INIT = [10, 10]
    VECTOR_SIZE_INIT = 10

    def __init__(self):
        self.learning_epochs = App.LEARNING_EPOCHS_INIT
        self.learning_rate = App.LEARNING_RATE_INIT
        self.architecture = App.ARCHITECTURE_INIT
        self.canvas_mode = App.CANVAS_MODE_TRAINING
        self.vector_size = App.VECTOR_SIZE_INIT

        self.canvas_trace = Trace()
        self.recording = False
        
        self.labels = set()
        self.training_traces = []
        self.training_labels = []
        self.classifier = None
        
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(App.WINDOW_SIZE, pygame.DOUBLEBUF | pygame.OPENGL)
        self.running = False

        self.query_show = False
        self.query_callback = None
        self.query_prompt = ""
        self.query_input = ""

        self.output = ""
        self.output_changed = False

        imgui.create_context()
        self.backend = imgui.integrations.pygame.PygameRenderer()
        self.io = imgui.get_io()
        self.io.display_size = App.WINDOW_SIZE

    def close(self):
        pygame.quit()

    def inform(self, text):
        self.output += text + "\n"
        self.output_changed = True

    def query(self, prompt, callback, default=""):
        self.query_input = default
        self.query_prompt = prompt
        self.query_callback = callback
        self.query_show = True

    def on_character_drawn(self, trace):
        if self.canvas_mode == App.CANVAS_MODE_TRAINING:
            def on_label_entered(label):
                self.training_traces.append(trace)
                self.training_labels.append(label)
                self.labels.add(label)
                trace.vectorize(2 * self.vector_size)
            self.query("Character label: ", on_label_entered)
        elif self.canvas_mode == App.CANVAS_MODE_INFERENCE and self.classifier != None:
            if len(trace.points) >= self.classifier.input_size // 2 + 1:
                input = trace.vectorize(2 * self.vector_size)
                prediction = self.classifier.classify(input)
                self.inform("Predicted output: %s" % prediction)
            else:
                self.canvas_trace = Trace()
                self.inform("Not enough points")
            

    def on_action_train(self):
        if len(self.training_traces) > 0:
            input_batch = [trace.vectorize(2 * self.vector_size) for trace in self.training_traces]
            if min([len(input) for input in input_batch]) < 2 * self.vector_size:
                self.inform("Not enough points")
                return
            
            self.inform("Training ...")
            architecture = [(neural.Dense, size) for size in self.architecture]
            self.classifier = neural.Classifier(self.vector_size * 2, architecture, self.labels)
            self.classifier.add_samples(input_batch, self.training_labels)
            self.classifier.train(self.learning_epochs, self.learning_rate)
            self.inform("Training complete")
            self.canvas_mode = App.CANVAS_MODE_INFERENCE
            self.canvas_trace = Trace()
        else:
            self.inform("Not enough training data")

    def on_action_clear_characters(self):
        self.labels = set()
        self.training_traces = []
        self.training_labels = []
        self.canvas_trace = Trace()
        self.inform("Characters cleared")
        self.canvas_mode = App.CANVAS_MODE_TRAINING

    def loop(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if not self.io.want_capture_mouse:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            self.canvas_trace = Trace()
                            self.recording = True
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            self.recording = False
                            if self.canvas_trace.is_valid():
                                self.on_character_drawn(self.canvas_trace)
                            else:
                                self.canvas_trace = Trace()
                self.backend.process_event(event)
            self.backend.process_inputs()

            imgui.new_frame()

            background_draw_list = imgui.get_background_draw_list()
            
            if self.canvas_mode == App.CANVAS_MODE_TRAINING:
                color = imgui.get_color_u32_rgba(*float_color(App.BACKGROUND_COLOR_TRAINING))
            elif self.canvas_mode == App.CANVAS_MODE_INFERENCE:
                color = imgui.get_color_u32_rgba(*float_color(App.BACKGROUND_COLOR_INFERENCE))
            background_draw_list.add_rect_filled(0, 0, *App.WINDOW_SIZE, color)

            if self.canvas_trace.is_valid():
                self.canvas_trace.draw(background_draw_list, App.LINE_COLOR, App.LINE_WIDTH)

            if imgui.begin_main_menu_bar().opened:
                if imgui.begin_menu("Configuration").opened:
                    
                    if imgui.menu_item("Input vector count: %i" % self.vector_size)[0]:
                        def callback(value):
                            try:
                                self.vector_size = int(value)
                            except:
                                self.inform("Invalid value")
                        self.query("Input vector count:", callback, str(self.vector_size))
                    if imgui.menu_item("Hidden layers shape: %s" % ",".join(map(str, self.architecture)))[0]:
                        def callback(value):
                            try:
                                self.architecture = list(map(int, value.split(",")))
                            except:
                                self.inform("Invalid value")
                        self.query("Hidden layers shape:", callback, ",".join(map(str, self.architecture)))
                    if imgui.menu_item("Epochs: %i" % self.learning_epochs)[0]:
                        def callback(value):
                            try:
                                self.learning_epochs = int(value)
                            except:
                                self.inform("Invalid value")
                        self.query("Epochs:", callback, str(self.learning_epochs))
                    if imgui.menu_item("Learning rate: %f" % self.learning_rate)[0]:
                        def callback(value):
                            try:
                                self.learning_rate = float(value)
                            except:
                                self.inform("Invalid value")
                        self.query("Learning rate:", callback, str(self.learning_rate))
                        
                    imgui.end_menu()
                
                if imgui.button("Train"):
                    self.on_action_train()

                if imgui.button("Clear characters"):
                    self.on_action_clear_characters()

                lines = [line.strip() for line in self.output.split("\n")]
                lines = [line for line in lines if line]
                if lines:
                    imgui.text("    ... " + lines[-1])
                
                imgui.end_main_menu_bar()

            if self.query_show:
                imgui.begin("##main", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_RESIZE)
                imgui.open_popup("##query")
                
                if imgui.begin_popup_modal("##query").opened:
                    imgui.text(self.query_prompt)
                    if imgui.is_window_appearing():
                        imgui.set_keyboard_focus_here()
                    enter, self.query_input = imgui.input_text("##query_input", self.query_input, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                    if imgui.button("Submit") or enter:
                        self.query_callback(self.query_input)
                        self.query_show = False
                        imgui.close_current_popup()
                    imgui.end_popup()
                    
                imgui.end()

            imgui.begin("Output")
            imgui.text(self.output)
            if self.output_changed:
                imgui.set_scroll_y(imgui.get_scroll_max_y() + 100)
                self.output_changed = False
            imgui.end()

            if self.recording:
                self.canvas_trace.add_point(pygame.mouse.get_pos())

            imgui.render()
            self.backend.render(imgui.get_draw_data())

            pygame.display.flip()
            self.clock.tick(App.WINDOW_FPS)



a = App()
a.loop()
a.close()
