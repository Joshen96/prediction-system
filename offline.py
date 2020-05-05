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
    "import numpy as np\n",
    "\n",
    "from pandas.io.parsers import read_csv\n",
    "\n",
    "model = tf.global_variables_initializer()\n",
    "\n",
    "data = read_csv('price data.csv', sep=',')\n",
    "\n",
    "xy = np.array(data, dtype=np.float32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 2.0100100e+07 -4.9000001e+00 -1.1000000e+01  8.9999998e-01\n",
      "   0.0000000e+00  2.1230000e+03]\n",
      " [ 2.0100102e+07 -3.0999999e+00 -5.5000000e+00  5.5000000e+00\n",
      "   8.0000001e-01  2.1230000e+03]\n",
      " [ 2.0100104e+07 -2.9000001e+00 -6.9000001e+00  1.4000000e+00\n",
      "   0.0000000e+00  2.1230000e+03]\n",
      " ...\n",
      " [ 2.0171228e+07  2.9000001e+00 -2.0999999e+00  8.0000000e+00\n",
      "   0.0000000e+00  2.9010000e+03]\n",
      " [ 2.0171230e+07  2.9000001e+00 -1.6000000e+00  7.0999999e+00\n",
      "   6.0000002e-01  2.9010000e+03]\n",
      " [ 2.0171232e+07  2.0999999e+00 -2.0000000e+00  5.8000002e+00\n",
      "   4.0000001e-01  2.9010000e+03]]\n"
     ]
    }
   ],
   "source": [
    "x_data=xy[:,1:-1]\n",
    "y_data=xy[:,[-1]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
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
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "hypothesis = tf.matmul(X, W) + b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "cost = tf.reduce_mean(tf.square(hypothesis - Y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)\n",
    "train = optimizer.minimize(cost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "sess = tf.Session()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "sess.run(tf.global_variables_initializer())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "학습된 모델을 저장했습니다.\n"
     ]
    }
   ],
   "source": [
    "saver = tf.train.Saver()\n",
    "save_path = saver.save(sess, \"./saved.cpkt\")\n",
    "print('학습된 모델을 저장했습니다.')"
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
