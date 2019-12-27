import gi

gi.require_version('Gdk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
gi.require_version('Gtk', '3.0')

from gi.repository import Gdk, Gtk, GdkPixbuf

class IdentifyWidget(Gtk.Layout):
    def __init__(self, filename, contacts, rois, name_change_cb):
        Gtk.Layout.__init__(self)

        self.__filename = filename
        self.__rois = rois
        self.__name_change_cb = name_change_cb
        self.__name_entry = None

        self.__pixbuf = GdkPixbuf.Pixbuf.new_from_file(self.__filename)

        self.__drawing_area = Gtk.DrawingArea()
        self.__drawing_area.connect('draw', self.__draw)

        event_box = Gtk.EventBox()
        event_box.add(self.__drawing_area)

        self.put(event_box, 0, 0)

        self.__entries = []

        for roi in self.__rois:
            entry = Gtk.Entry()

            entry.connect('focus-out-event', lambda widget, event, roi=roi: self.__entry_focus_out(widget, roi))

            if roi.contact and roi.contact in contacts:
                entry.set_text(contacts[roi.contact])

            self.__entries.append(entry)
            self.put(entry, 0, 0)

        def on_widget_resize(widget, alloc, *args):
            event_box.set_size_request(alloc.width, alloc.height)
            return False

        # Scale event box to entire widget
        self.connect_after('size-allocate', on_widget_resize)

        def place_entries(*args):
            for roi, entry in zip(self.__rois, self.__entries):
                rx, ry, rw, rh = self.__roi_rect(roi)
                label_x, label_y = self.__coord_img_to_win(rx, ry+rh)
                self.move(entry, label_x, 5+label_y)

            return False

        self.__drawing_area.connect_after('size-allocate', place_entries)

    def __entry_focus_out(self, entry, roi):
        text = entry.get_text()

        if text:
            self.__name_change_cb(roi, text)

        return False

    def __coord_win_to_img(self, x, y):
        width = self.__drawing_area.get_allocated_width()
        height = self.__drawing_area.get_allocated_height()
        img_width, img_height = self.__pixbuf.get_width(), self.__pixbuf.get_height()

        resize_ratio = min(width/img_width, height/img_height)
        dx = (width-img_width*resize_ratio)/2
        dy = (height-img_height*resize_ratio)/2

        return ((x-dx)/resize_ratio, (y-dy)/resize_ratio)

    def __coord_img_to_win(self, x, y):
        width = self.__drawing_area.get_allocated_width()
        height = self.__drawing_area.get_allocated_height()
        img_width, img_height = self.__pixbuf.get_width(), self.__pixbuf.get_height()

        resize_ratio = min(width/img_width, height/img_height)
        dx = (width-img_width*resize_ratio)/2
        dy = (height-img_height*resize_ratio)/2

        return (dx+x*resize_ratio, dy+y*resize_ratio)

    def __roi_rect(self, roi):
        img_width, img_height = self.__pixbuf.get_width(), self.__pixbuf.get_height()
        return (img_width*roi.x, img_height*roi.y, img_width*roi.w, img_height*roi.h)

    def __roi_contains(self, roi, x, y):
        rx, ry, rw, rh = self.__roi_rect(roi)
        return x >= rx and y >= ry and x <= (rx+rw) and y <= (ry+rh)


    def __draw(self, widget, cr):
        context = widget.get_style_context()
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()

        Gtk.render_background(context, cr, 0, 0, width, height)

        img_width, img_height = self.__pixbuf.get_width(), self.__pixbuf.get_height()

        resize_ratio = min(width/img_width, height/img_height)
        dx = (width-img_width*resize_ratio)/2
        dy = (height-img_height*resize_ratio)/2

        cr.save()
        cr.translate(dx, dy)
        cr.scale(resize_ratio, resize_ratio)

        Gdk.cairo_set_source_pixbuf(cr, self.__pixbuf, 0, 0)
        cr.paint()

        cr.set_line_width(3/resize_ratio)
        cr.set_source_rgb(.2, .2, .2)

        for roi in self.__rois:
            cr.rectangle(img_width*roi.x, img_height*roi.y, img_width*roi.w, img_height*roi.h)
            cr.stroke()

        cr.restore()

        return False


def identify_window(filename, contacts, rois, name_change_cb):
    win = Gtk.Window()
    win.connect('destroy', Gtk.main_quit)
    win.add(IdentifyWidget(filename, contacts, rois, name_change_cb))
    win.show_all()
    Gtk.main()
