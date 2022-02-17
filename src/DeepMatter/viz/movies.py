import glob as glob
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def make_movie(movie_name,
               input_folder,
               output_folder,
               file_format,
               fps,
               output_format='mp4',
               reverse=False):
    """

    :param movie_name: Name of movie
    :type movie_name: string
    :param input_folder: folder where image series is located
    :type input_folder: string
    :param output_folder: location to save movie
    :type output_folder: string
    :param file_format: sets the file format to import
    :type file_format: string
    :param fps: frames-per-second
    :type fps: int
    :param output_format: sets the format which the movie will be saved as
    :type output_format: string
    :param reverse: selects if the movie will be shown in reverse appended to end
    :type reverse: boolean

    """

    # searches the folder and finds the files
    file_list = glob.glob('./' + input_folder + '/*.' + file_format)
    #     print(input_folder)
    #     print(file_list)

    # Sorts the files by number makes 2 lists to go forward and back
    list.sort(file_list)
    file_list_rev = glob.glob('./' + input_folder + '/*.' + file_format)
    list.sort(file_list_rev, reverse=True)

    # combines the file list if including the reverse
    if reverse:
        new_list = file_list + file_list_rev
    else:
        new_list = file_list
    #        print(new_list)

    if output_format == 'gif':
        # makes an animated gif from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_gif(output_folder + '/{}.gif'.format(movie_name), fps=fps)
    else:

        # makes and mp4 from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_videofile(output_folder + '/{}.mp4'.format(movie_name), fps=fps)

def make_movie(movie_name, input_folder, output_folder, file_format,
                            fps, output_format = 'mp4', reverse = False):
    """
    function that makes the movie of the images data

    :param movie_name: name of the movie
    :type movie_name: string
    :param input_folder: folder where the image series is located
    :type input_folder: string
    :param output_folder: folder where the movie will be saved
    :type output_folder: string
    :param file_format: sets the format of the files to import
    :type file_format: string
    :param fps: frames per second
    :type fps: numpy, int
    :param output_format: sets the format for the output file
                          supported types .mp4 and gif
                          animated gif create large files
    :type output_format: string (, optional)
    :param reverse: sets if the movie will be one way of there and back
    :type reverse: bool  (, optional)

    """


    # searches the folder and finds the files
    file_list = glob.glob(input_folder + '/*.' + file_format)

    # Sorts the files by number makes 2 lists to go forward and back
    list.sort(file_list)
    file_list_rev = glob.glob(input_folder + '/*.' + file_format)
    list.sort(file_list_rev,reverse=True)

    # combines the file list if including the reverse
    if reverse:
        new_list = file_list + file_list_rev
    else:
        new_list = file_list


    if output_format == 'gif':
        # makes an animated gif from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_gif(output_folder + '/{}.gif'.format(movie_name), fps=fps)
    else:
        # makes and mp4 from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_videofile(output_folder + '/{}.mp4'.format(movie_name), fps=fps)
