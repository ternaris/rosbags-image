Image
=====

The :py:mod:`rosbags.image` package provides ROS image message tools aiming to provide the functionality delivered by cv_bridge in the native ros stack.

Reading CompressedImage
-----------------------
The function :py:func:`compressed_image_to_cvimage <rosbags.image.compressed_image_to_cvimage>` converts ``CompressedImage`` message instances to OpenCV images:

.. code-block:: python

   from rosbags.image import compressed_image_to_cvimage

   # msg is a CompressedImage message instance
   msg = ...

   # get image in source color space
   img = compressed_image_to_cvimage(msg)

   # get image and convert to specific color space
   img = compressed_image_to_cvimage(msg, 'mono8')


Reading Image
-------------
The function :py:func:`image_to_cvimage <rosbags.image.image_to_cvimage>` converts ``Image`` message instances to OpenCV images:

.. code-block:: python

   from rosbags.image import image_to_cvimage

   # msg is a Image message instance
   msg = ...

   # get image in source color space
   img = image_to_cvimage(msg)

   # get image and convert to specific color space
   img = image_to_cvimage(msg, 'mono8')


Autodetect message type
-----------------------
For use cases where the incoming message instances can be ``CompressedImage`` or ``Image`` the helper function :py:func:`message_to_cvimage <rosbags.image.message_to_cvimage>` autodetects the incoming type and returns OpenCV images:

.. code-block:: python

   from rosbags.image import message_to_cvimage

   # msg is a CompressedImage or Image message instance
   msg = ...

   # get image in source color space
   img = message_to_cvimage(msg)

   # get image and convert to specific color space
   img = message_to_cvimage(msg, 'mono8')