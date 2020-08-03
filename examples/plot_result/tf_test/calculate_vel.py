import  tensorflow as tf
import matplotlib.pyplot as plt

with tf.device('/GPU:0'):
    first = tf.constant([8.000,8.000,8.000,8.000,8.000], dtype='float32', shape=[1, 5], name='a')
    last = tf.constant([8.387,8.826,9.238,9.630,10.144], dtype='float32', shape=[1, 5], name='b')
    sec = tf.constant([1,2,3,4,5], dtype='float32', shape=[1, 5], name='s')
    frame = tf.constant([46,92,135,176,230], dtype='float32', shape=[1, 5], name='f')

length = tf.math.abs(tf.math.subtract(first, last))
fps = tf.math.divide(frame,sec)
vel = tf.math.divide(length,sec)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print("Length:",sess.run(length))
print("Velocity:",sess.run(vel))
print("FPS:",sess.run(fps))

plt.figure(1, figsize=(18, 10))
plt.subplot(1,2,1)
plt.title('Velocity')
plt.xlabel("Time"); plt.ylabel("Velocity")
plt.plot(sess.run(sec), sess.run(vel), 'r*', linewidth=2); plt.grid(True)
plt.subplot(1,2,2)
plt.title('FPS')
plt.xlabel("Sec"); plt.ylabel("Frame")
plt.plot(sess.run(sec), sess.run(fps), 'bo', linewidth=2); plt.grid(True)
plt.tight_layout()
plt.savefig('velocity_frame.jpg')
plt.show()
