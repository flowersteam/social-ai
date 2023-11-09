import sys
import numpy as np

# Only ask users to install matplotlib if they actually need it
try:
    import matplotlib.pyplot as plt
except:
    print('To display the environment in a window, please install matplotlib, eg:')
    print('pip3 install --user matplotlib')
    sys.exit(-1)

class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title, figsize=(3, 3)):
        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots(
            # figsize=(10, 5),
            figsize=figsize,
        )

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.xaxis.set_ticks_position('none')
        self.ax.yaxis.set_ticks_position('none')
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # list of text handles
        self.txt_handles = []

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        # plt.pause(0.001)

    def set_caption(self, text, relevant_set=None):
        """
        Set/update the caption text below the image
        """

        # plt.xlabel(text)
        # text = "All utterances:\n\n"+text
        lines = text.split("\n")

        if len(lines) > 8:
            lines = ["..."]+lines[-8:]

        text = "\n".join(lines)

        if hasattr(self, "caption"):
            self.caption.set_text(text)
        else:
            # self.caption = plt.text(400, 250, text, ha="left",wrap=True)
            self.caption = plt.text(330, 250, text, ha="left", wrap=True)

        if relevant_set is not None:
            # if a line in the text has one of these strings it will be put in the relevant set

            relevant_lines = ["Relevant utterances:\n"] + [
                l for l in text.rsplit("\n") if any([r in l for r in relevant_set])
            ] + ["\n"]
            relevant_text = "\n".join(relevant_lines)


            if hasattr(self, "relevant_caption"):
                self.relevant_caption.set_text(relevant_text)
            else:
                self.relevant_caption = plt.text(-200, 250, relevant_text, ha="left")


    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()

    def add_text(self, *args, **kwargs):

        kwargs['transform'] = self.ax.transAxes
        self.txt_handles.append(self.ax.text(*args, **kwargs))

    def clear_text(self):

        if len(self.txt_handles) > 0:
            while len(self.txt_handles) > 0:
                self.txt_handles.pop().remove()
