#!/usr/bin/env python3
import socket
import hashlib
import os.path

def get_xml_from_alpino(sentence):
    # returned cached xml file
    filename = get_filename_for_sentence(sentence)
    if os.path.isfile(filename):
        return get_xml_from_file(filename)

    # always send with newline at the end
    if sentence[-1] != "\n":
        sentence = sentence + "\n"

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("alpino.dionbosschieter.nl", 42424))
    s.sendall(str.encode(sentence))
    all_data = []
    while True:
            data = s.recv(4096)
            if data == b"":
                break
            all_data.append(data)

    s.close()

    xml = b"".join(all_data).decode()
    write_xml_to_file(xml, filename)

    return xml


def get_xml_from_file(filename):
    with open(filename, "r") as f:
        return f.read()


def get_filename_for_sentence(sentence):
    hash = hashlib.sha1(str.encode(sentence))
    return "xmlcache/" + hash.hexdigest() + ".xml"


def write_xml_to_file(xml, filename):
    with open(filename, "w") as f:
        f.write(xml)


def get_starting_node(tree):
    """
    Return the root node after the top node, while skipping all the punctuation
    """
    root_node = False
    # get top node
    for child in tree.getroot():
        if child.get("rel") == "top":
            root_node = child

    # get start of sentence, skipping punctuation
    for child in root_node:
        if child.get("begin") == "0":
            root_node = child

    return root_node
