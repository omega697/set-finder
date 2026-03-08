package com.guywithburrito.setfinder

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.guywithburrito.setfinder.ui.SetFinderView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import com.guywithburrito.setfinder.ui.SetFinderTheme
import org.opencv.android.OpenCVLoader

import com.guywithburrito.setfinder.ui.SetFinderNavGraph

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!OpenCVLoader.initDebug()) {
            Log.e("MainActivity", "Unable to load OpenCV!")
        } else {
            Log.d("MainActivity", "OpenCV loaded Successfully!")
        }
        setContent {
            SetFinderTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    SetFinderNavGraph()
                }
            }
        }
    }
    }