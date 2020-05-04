{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Python_Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\compat\\v2_compat.py:88: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "non-resource variables are not supported in the long term\n"
     ]
    }
   ],
   "source": [
    "import tensorflow.compat.v1 as tf\n",
    "tf.disable_v2_behavior()\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = tf.placeholder(tf.float32, shape=[None, 4])\n",
    "Y = tf.placeholder(tf.float32, shape=[None, 1])\n",
    "W = tf.Variable(tf.random_normal([4,1]), name=\"weight\")\n",
    "b = tf.Variable(tf.random_normal([1]), name=\"bias\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "hypothesis = tf.matmul(X, W) + b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "saver = tf.train.Saver()\n",
    "model = tf.global_variables_initializer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "평균 온도: 15.5\n",
      "최저 온도: 3.5\n",
      "최고 온도: 20.5\n",
      "강수량: 5\n"
     ]
    }
   ],
   "source": [
    "avg_temp = float(input('평균 온도: '))\n",
    "min_temp = float(input('최저 온도: '))\n",
    "max_temp = float(input('최고 온도: '))\n",
    "rain_fall = float(input('강수량: '))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Restoring parameters from ./saved.cpkt\n",
      "[37.722603]\n"
     ]
    }
   ],
   "source": [
    "with tf.Session() as sess:\n",
    "    sess.run(model)\n",
    "    \n",
    "    save_path=\"./saved.cpkt\"\n",
    "    saver.restore(sess, save_path)\n",
    "    \n",
    "    data=((avg_temp, min_temp, max_temp, rain_fall), )\n",
    "    arr = np.array(data, dtype=np.float32)\n",
    "    \n",
    "    x_data = arr[0:4]\n",
    "    dict = sess.run(hypothesis, feed_dict={X: x_data})\n",
    "    print(dict[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
