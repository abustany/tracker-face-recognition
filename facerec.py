#!/usr/bin/env python3

from collections import namedtuple
import io
import os
import os.path
import sqlite3
import urllib.parse

import click
import gi
import numpy as np

gi.require_version('GLib', '2.0')
gi.require_version('Tracker', '2.0')

from gi.repository import GLib, Tracker

import gui

class EmbeddingStore:
    def __init__(self, data_dir):
        os.makedirs(data_dir, mode=0o700, exist_ok=True)
        self.conn = sqlite3.connect(os.path.join(data_dir, 'index.db'))
        self.__create_schema()

    def __create_schema(self):
        schema = [
            '''CREATE TABLE IF NOT EXISTS known_files (uri TEXT NOT NULL, PRIMARY KEY(uri))''',
            '''CREATE TABLE IF NOT EXISTS roi_embeddings (uri TEXT NOT NULL, embedding BLOB, PRIMARY KEY(uri))'''
        ]

        for s in schema:
            self.conn.execute(s)
        self.conn.commit()

    def has(self, file_uri):
        c = self.conn.execute('SELECT EXISTS (SELECT 1 FROM known_files WHERE uri = ?)', (file_uri,))
        return bool(c.fetchone()[0])

    def save_indexed(self, file_uri):
        with self.conn:
            self.conn.execute('INSERT INTO known_files VALUES (?)', (file_uri,))

    def save_embedding(self, roi_uri, embedding):
        embedding_data = io.BytesIO()
        np.save(embedding_data, embedding)
        self.conn.execute('INSERT INTO roi_embeddings VALUES (?, ?)', (roi_uri, embedding_data.getvalue()))

    def get_embeddings(self, roi_uris):
        uris = []
        embeddings = []

        for row in self.conn.execute('SELECT uri, embedding FROM roi_embeddings WHERE uri IN (%s)' % ','.join(['?']*len(roi_uris)), roi_uris):
            uris.append(row[0])
            embeddings.append(np.load(io.BytesIO(row[1])))

        return (uris, np.concatenate([embeddings]))


    def clear(self):
        with self.conn:
            self.conn.execute('DELETE FROM known_files')
            self.conn.execute('DELETE FROM roi_embeddings')


def make_embedding_store():
    return EmbeddingStore(os.path.join(GLib.get_user_data_dir(), 'tracker-facerec'))


def make_tracker_conn():
    return Tracker.SparqlConnection.get()


@click.group()
def cli():
    pass


def list_pictures(conn, in_directory):
    """
    Returns a generator over the picture URIs in the given directory.
    """

    cursor = conn.query('SELECT ?url {?f a nfo:Image; nie:url ?url . FILTER(strstarts(?url, "%s"))}' % uri_from_path(in_directory))
    uris = []

    try:
        while cursor.next():
            uri = cursor.get_string(0)[0]

            if uri is None:
                continue

            yield uri
    finally:
        cursor.close()

    return uris


def path_from_uri(uri):
    return urllib.parse.unquote(urllib.parse.urlparse(uri).path)


def uri_from_path(path):
    return 'file://' + urllib.parse.quote(path, safe='/!$&\'()*+,;=:@')


def default_pictures_dir():
    return GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_PICTURES)


@cli.command('list-pictures')
@click.option('--pictures-dir', default=default_pictures_dir(), help='Directory where pictures are', show_default=True)
def cmd_list_pictures(pictures_dir):
    """
    Lists the pictures in the index
    """

    conn = make_tracker_conn()

    print('\n'.join([path_from_uri(f) for f in list_pictures(conn, pictures_dir)]))


def get_file_uri(conn, filename):
    cursor = conn.query('SELECT ?f {?f nie:url "%s"}' % uri_from_path(filename))

    uri = None

    try:
        while cursor.next():
            uri = cursor.get_string(0)[0]
            break
    finally:
        cursor.close()

    return uri


def get_identified_rois(conn):
    cursor = conn.query('''SELECT ?r {
        ?r a nfo:RegionOfInterest;
        nfo:regionOfInterestType nfo:roi-content-face;
        nfo:roiRefersTo ?x
    }''')

    try:
        while cursor.next():
            yield cursor.get_string(0)[0]
    finally:
        cursor.close()


def index_picture(conn, store, filename):
    file_uri = get_file_uri(conn, filename)

    if not file_uri:
        raise Exception('File is not in index: %s' % filename)

    if store.has(file_uri):
        print('File is already indexed: %s' % filename)
        return

    import cv2
    import face_recognition

    try:
        with open(filename, 'rb') as fd:
            img = face_recognition.load_image_file(fd)
    except IOError as e:
        print('Error opening %s: %s' % (filename, e))
        store.save_indexed(file_uri)
        return

    img_width = img.shape[1]
    img_height = img.shape[0]
    resize_ratio = min(1, 800/min(img_width, img_height))
    img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
    face_locations = face_recognition.face_locations(img)
    face_embeddings = face_recognition.face_encodings(img, face_locations)

    if len(face_locations) == 0:
        print('No faces detected: %s' % filename)
        store.save_indexed(file_uri)
        return

    region_uris = []

    for (top, right, bottom, left) in face_locations:
        x = left/resize_ratio/img_width
        y = top/resize_ratio/img_height
        w = (right-left)/resize_ratio/img_width
        h = (bottom-top)/resize_ratio/img_height
        res = conn.update_blank('''INSERT {
            _:region a nfo:RegionOfInterest ;
                nfo:regionOfInterestX %f ;
                nfo:regionOfInterestY %f ;
                nfo:regionOfInterestWidth %f ;
                nfo:regionOfInterestHeight %f ;
                nfo:regionOfInterestType nfo:roi-content-face .
            <%s> nfo:hasRegionOfInterest _:region
            }''' % (x, y, w, h, file_uri), 0, None)
        region_uris.append(res.unpack()[0][0]['region'])

    for roi_uri, embedding in zip(region_uris, face_embeddings):
        store.save_embedding(roi_uri, embedding)

    store.save_indexed(file_uri)

    print('Inserted regions %s' % ', '.join(region_uris))


@cli.command('index-picture')
@click.argument('filename', nargs=-1)
def cmd_index_picture(filename):
    """
    Index the faces for the given picture.
    """

    conn = make_tracker_conn()
    store = make_embedding_store()

    for f in filename:
        index_picture(conn, store, f)


ROI = namedtuple('ROI', 'uri x y w h contact')


def get_face_rois(conn, file_uri):
    cursor = conn.query('''SELECT ?r ?x ?y ?w ?h ?contact {
            <%s> nfo:hasRegionOfInterest ?r .
            ?r nfo:regionOfInterestType nfo:roi-content-face ;
                nfo:regionOfInterestX ?x ;
                nfo:regionOfInterestY ?y ;
                nfo:regionOfInterestWidth ?w ;
                nfo:regionOfInterestHeight ?h .
            OPTIONAL { ?r nfo:roiRefersTo ?contact }
            }''' % file_uri)

    try:
        while cursor.next():
            yield ROI(
                uri=cursor.get_string(0)[0],
                x=cursor.get_double(1),
                y=cursor.get_double(2),
                w=cursor.get_double(3),
                h=cursor.get_double(4),
                contact=cursor.get_string(5)[0],
            )
    finally:
        cursor.close()


def get_roi(conn, uri):
    cursor = conn.query('''SELECT ?x ?y ?w ?h ?contact {
            <%(uri)s> nfo:regionOfInterestType nfo:roi-content-face ;
                nfo:regionOfInterestX ?x ;
                nfo:regionOfInterestY ?y ;
                nfo:regionOfInterestWidth ?w ;
                nfo:regionOfInterestHeight ?h .
            OPTIONAL { <%(uri)s> nfo:roiRefersTo ?contact }
            }''' % {'uri': uri})

    roi = None

    try:
        while cursor.next():
            roi = ROI(
                uri=uri,
                x=cursor.get_double(0),
                y=cursor.get_double(1),
                w=cursor.get_double(2),
                h=cursor.get_double(3),
                contact=cursor.get_string(4)[0],
            )
            break
    finally:
        cursor.close()

    return roi


def get_contact_name(conn, uri):
    name = None

    cursor = conn.query('SELECT ?name {<%s> nco:fullname ?name}' % uri)
    if cursor.next():
        name = cursor.get_string(0)[0]

    cursor.close()

    if not name:
        raise Exception('No contact with URI %s' % uri)

    return name


def get_contact_with_name(conn, name):
    escaped_name = Tracker.sparql_escape_string(name)
    contact = None

    cursor = conn.query('SELECT ?c {?c a nco:PersonContact; nco:fullname "%s"}' % escaped_name)
    if cursor.next():
        contact = cursor.get_string(0)[0]

    cursor.close()

    if not contact:
        # No contact found, create a new one
        res = conn.update_blank('INSERT { _:c a nco:PersonContact; nco:fullname "%s" }' % escaped_name, 0, None)
        contact = res.unpack()[0][0]['c']

    return contact


def get_contacts(conn):
    contacts = {}
    cursor = conn.query('SELECT ?c ?name {?c a nco:PersonContact; nco:fullname ?name}')

    while cursor.next():
        contacts[cursor.get_string(0)[0]] = cursor.get_string(1)[0]

    cursor.close()

    return contacts


def autoidentify_picture(conn, store, filename):
    import face_recognition

    known_roi_uris, known_embeddings = store.get_embeddings(list(get_identified_rois(conn)))
    matches = {}

    if len(known_roi_uris) == 0:
        return matches

    file_uri = get_file_uri(conn, filename)
    file_roi_uris, file_embeddings = store.get_embeddings(list([roi.uri for roi in get_face_rois(conn, file_uri)]))

    if len(file_roi_uris) == 0:
        return matches

    for roi_uri, embedding in zip(file_roi_uris, file_embeddings):
        if roi_uri in known_roi_uris:
            continue

        face_distances = face_recognition.face_distance(known_embeddings, embedding)
        best_match_index = np.argmin(face_distances)
        distance = face_distances[best_match_index]

        if distance > .6:
            continue

        roi = get_roi(conn, known_roi_uris[best_match_index])
        print('Autodetected %s (distance: %f) ' % (get_contact_name(conn, roi.contact), face_distances[best_match_index]))
        matches[roi_uri] = roi.contact

    return matches


def associate_roi_to_contact(conn, roi_uri, contact_uri):
        conn.update('''
            DELETE {<%(roi_uri)s> nfo:roiRefersTo ?x} WHERE {<%(roi_uri)s> nfo:roiRefersTo ?x}
            INSERT {<%(roi_uri)s> nfo:roiRefersTo <%(contact_uri)s>}
        ''' % {'roi_uri': roi_uri, 'contact_uri': contact_uri}, 0, None)


def identify_picture(conn, filename, automatches={}):
    file_uri = get_file_uri(conn, filename)

    changed_rois = {}

    def name_change_cb(roi, name):
        changed_rois[roi.uri] = name

    rois = list(get_face_rois(conn, file_uri))

    for i, roi in enumerate(rois):
        if roi.uri in automatches:
            rois[i] = roi._replace(contact = automatches[roi.uri])

    gui.identify_window(filename, get_contacts(conn), rois, name_change_cb)

    for uri, name in changed_rois.items():
        print('Associating ROI %s to %s' % (uri, name))
        contact_uri = get_contact_with_name(conn, name)
        associate_roi_to_contact(conn, uri, contact_uri)

    for uri, contact_uri in automatches.items():
        if uri in changed_rois:
            continue

        print('Associating ROI %s to %s' % (uri, get_contact_name(conn, contact_uri)))
        associate_roi_to_contact(conn, uri, contact_uri)


@cli.command('identify-picture')
@click.argument('filename')
def cmd_identify_picture(filename):
    """
    Open the face identification GUI for the given picture.
    """

    conn = make_tracker_conn()
    store = make_embedding_store()
    matches = autoidentify_picture(conn, store, filename)
    identify_picture(conn, filename, matches)


def get_next_unidentified_picture_uri(conn):
    cursor = conn.query('SELECT ?url {?file nie:url ?url; nfo:hasRegionOfInterest ?roi . FILTER(NOT EXISTS {?roi nfo:roiRefersTo ?something})} LIMIT 1')
    uri = None

    try:
        while cursor.next():
            uri = cursor.get_string(0)[0]
            break
    finally:
        cursor.close()

    return uri


@cli.command('identify-next-picture')
def cmd_identify_next_picture():
    """
    Opens the identification GUI for the next picture having an unidentified ROI
    """

    conn = make_tracker_conn()
    uri = get_next_unidentified_picture_uri(conn)

    if not uri:
        print('All ROIs are identified. Good job!')
        return

    path = path_from_uri(uri)
    print('Identifying picture %s' % path)
    cmd_identify_picture([path])


def list_rois(conn):
    cursor = conn.query('SELECT ?r ?name {?r nfo:roiRefersTo ?c . ?c a nco:PersonContact; nco:fullname ?name}')

    while cursor.next():
        print('ROI %s refers to %s' % (cursor.get_string(0)[0], cursor.get_string(1)[0]))

    cursor.close()


@cli.command('list-rois')
def cmd_list_rois():
    """
    Lists all ROIs
    """
    list_rois(make_tracker_conn())


def show_roi(conn, uri):
    import cv2

    cursor = conn.query('SELECT ?url {?f nfo:hasRegionOfInterest <%s>; nie:url ?url}' % uri)
    file_uri = None

    while cursor.next():
        file_uri = cursor.get_string(0)[0]
        break

    cursor.close()

    if not file_uri:
        raise Exception('ROI is not linked to any file')

    roi = get_roi(conn, uri)
    img = cv2.imread(path_from_uri(file_uri))
    width, height = img.shape[1], img.shape[0]
    top, right, bottom, left = int(roi.y*height), int((roi.x+roi.w)*width), int((roi.y+roi.h)*height), int(roi.x*width)
    cv2.imshow(roi.uri, img[top:bottom, left:right])
    cv2.waitKey()


@cli.command('show-roi')
@click.argument('uri')
def cmd_show_roi(uri):
    """
    Shows the given ROI graphically.
    """

    show_roi(make_tracker_conn(), uri)


def index_all_pictures(conn, store, pictures_dir):
    """
    Indexes all pictures in the index
    """

    for uri in list_pictures(conn, pictures_dir):
        if store.has(uri):
            continue

        path = path_from_uri(uri)
        print('Indexing %s' % path)
        index_picture(conn, store, path)


@cli.command('index-all-pictures')
@click.option('--pictures-dir', default=default_pictures_dir(), help='Directory where pictures are', show_default=True)
def cmd_index_all_pictures(pictures_dir):
    index_all_pictures(make_tracker_conn(), make_embedding_store(), pictures_dir)


@cli.command()
def clear_all_regions():
    """
    Deletes all regions of interest from Tracker.
    """

    conn = make_tracker_conn()
    cursor = conn.query('SELECT ?f ?r {?f nfo:hasRegionOfInterest ?r}')

    try:
        while cursor.next():
            file_uri = cursor.get_string(0)[0]
            region_uri = cursor.get_string(1)[0]
            conn.update('DELETE {<%s> nfo:hasRegionOfInterest <%s>}' % (file_uri, region_uri), 0, None)
            conn.update('DELETE {<%s> a rdfs:Resource}' % (region_uri), 0, None)
    finally:
        cursor.close()

    make_embedding_store().clear()

if __name__ == '__main__':
    cli()
