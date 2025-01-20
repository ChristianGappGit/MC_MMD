#from
#https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def colorize_words(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.colormaps.get_cmap('RdBu')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string

def colorize_tokens(tokens, color_array, color_map='RdBu', mapping_range=[0,1], cbar_length=100):
    """
    tokens is a list of tokens
    note that tokens representing the end of a word should include a space at its end
    color_array is an array of numbers between 0 and 1 of length equal to tokens
    """ 
    cmap = matplotlib.colormaps.get_cmap(color_map)
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for token, color in zip(tokens, color_array):
        assert mapping_range[0] < mapping_range[1] and mapping_range[0] >= 0 and mapping_range[1] <= 1
        color = mapping_range[0] + color * (mapping_range[1] - mapping_range[0])
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, token)
    return colored_string, color_bar(cmap, mapping_range, length=cbar_length)

def color_bar(cmap, mapping_range=[0,1], length=100):
    """
    cmap is a matplotlib colormap
    mapping_range is a list of two numbers between 0 and 1
    """
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colorbar_string = ''
    for i in range(length):
        color = matplotlib.colors.rgb2hex(cmap(mapping_range[0] + i/length * (mapping_range[1] - mapping_range[0]))[:3])
        colorbar_string += template.format(color, '&nbsp' + '&nbsp')
    return colorbar_string

def colorize_tokens2(tokens, color_array, token_concat='##', 
                     skip_tokens=None, color_map='turbo', mapping_range=[0,1], 
                     max_chars_per_row=60,):
    """
    tokens is a list of text tokens
    note that tokens including "##" delete the former space, so that the token x can be concatenated with the former token x-1
    tokens without "##" represent the start of a new word, so that a space is added at the beginning of the token
    color_array is an array of numbers between 0 and 1 of length equal to tokens
    """
    if token_concat == None:
        raise ValueError('token_concat must be set to a string, e.g. "##"')
    cmap = matplotlib.colormaps.get_cmap(color_map)
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    num_chars_in_row = 0
    num_letters_in_word = 0
    for k, (token, color) in enumerate(zip(tokens, color_array)):
        if skip_tokens is not None and token in skip_tokens:
            continue
        assert mapping_range[0] < mapping_range[1] and mapping_range[0] >= 0 and mapping_range[1] <= 1
        color = mapping_range[0] + color * (mapping_range[1] - mapping_range[0])
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        if token.startswith(token_concat):  #concatenate with former token
            colored_string += template.format(color, token[len(token_concat):])
            num_chars_in_row += len(token[len(token_concat):])
            num_letters_in_word += len(token[len(token_concat):])
            #handle case where num_chars_in_row > max_chars_per_row
            #insert a line break before the whole word (not just the token), so that the whole word is moved to the next line
            #find first '&nbsp' before the token, delete the space and insert <br> before it.
            if max_chars_per_row is not None and num_chars_in_row > max_chars_per_row:
                colored_string = colored_string[::-1]
                index = colored_string.find('psbn&')
                #correct index as we reversed the string
                index = len(colored_string) - index - 5
                colored_string = colored_string[::-1] #reverse again
                last_template_word = colored_string[index+5:]
                colored_string = colored_string[:index] + '<br>' + last_template_word
                #set num_chars_in_row to lenght of last word
                print("last_template_word", last_template_word)
                num_chars_in_row = num_letters_in_word
        elif k == 0:    #first token
            colored_string += template.format(color, token)
            num_chars_in_row = len(token)
            num_letters_in_word = len(token)
        else: #new word
            num_chars_in_row += len(token)
            num_letters_in_word = len(token)
            if max_chars_per_row is not None and num_chars_in_row > max_chars_per_row:
                colored_string += template.format(color, '<br>' + token)
                num_chars_in_row = len(token) #new line with first token, reset num_chars_in_row
            else: # add space between tokens
                colored_string += template.format(color, '&nbsp' + token) #add space at beginning of new word
                num_chars_in_row += 1 #add space
        print("num_chars_in_row", num_chars_in_row)
    return colored_string, color_bar(cmap, mapping_range, length=max_chars_per_row)

#example usage:
def example_words():
    words = 'The quick brown fox jumps over the lazy dog'.split()
    color_array = np.random.rand(len(words))
    s = colorize_words(words, color_array)

    # to display in ipython notebook
    from IPython.display import display, HTML
    display(HTML(s))

    # or simply save in an html file and open in browser
    with open('colorize_words.html', 'w') as f:
        f.write(s)

def example_tokens():
    tokens = ['Th', 'e ', 'qu', 'ick ', 'bro', 'wn ', 'fo', 'x ', 'jum', 'ps ', 'ov',
               'er ', 'th', 'e ', 'la', 'zy ', 'do', 'g']
    color_array = np.random.rand(len(tokens))
    s = colorize_tokens(tokens, color_array)

    # to display in ipython notebook
    from IPython.display import display, HTML
    display(HTML(s))

    # or simply save in an html file and open in browser
    with open('colorize_tokens.html', 'w') as f:
        f.write(s)

def example_tokens2():
    tokens = ['Th', '##e', 'qu', '##ick', 'bro', '##wn', 'fo', '##x', 'jum', '##ps', 'ov',
               '##er', 'th', '##e', 'la', '##zy', 'do', '##g']
    color_array = np.random.rand(len(tokens))
    print("color_array", color_array)
    s = colorize_tokens2(tokens, color_array) #interpret tokens as words this time

    # to display in ipython notebook
    from IPython.display import display, HTML
    display(HTML(s))

    # or simply save in an html file and open in browser
    with open('colorize_tokens.html', 'w') as f:
        f.write(s)