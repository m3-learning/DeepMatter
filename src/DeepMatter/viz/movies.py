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
    Function which makes movies from an image series

    Args:
        movie_name: Name of movie
        input_folder: folder where image series is located
        output_folder: location to save movie
        file_format: sets the file format to import
        fps: frames-per-second
        output_format: sets the format which the movie will be saved as
        reverse: selects if the movie will be shown in reverse appended to end

    Returns:

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
