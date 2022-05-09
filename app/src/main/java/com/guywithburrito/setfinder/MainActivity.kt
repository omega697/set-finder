package com.guywithburrito.setfinder

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionRequired
import com.google.accompanist.permissions.rememberPermissionState
import com.guywithburrito.setfinder.ui.theme.SetFinderTheme
import org.opencv.android.OpenCVLoader

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
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    PermissionPreview()
                }
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun PermissionPreview() {
    val cameraPermissionState = rememberPermissionState(android.Manifest.permission.CAMERA)
    PermissionRequired(
        permissionState = cameraPermissionState,
        permissionNotGrantedContent = {
            Button(onClick = {
                cameraPermissionState.launchPermissionRequest()
            }) {
                Text("Grant permission")
            }
        },
        permissionNotAvailableContent = {
            Text("Sorry, the Camera permission isn't available!")
        }) {
        SetFinderView()
    }
}