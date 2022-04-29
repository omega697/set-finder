package com.guywithburrito.setfinder

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.material.Button
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionRequired
import com.guywithburrito.setfinder.ui.theme.SetFinderTheme
import com.google.accompanist.permissions.rememberPermissionState

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SetFinderTheme {
                // A surface container using the 'background' color from the theme
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colors.background) {
                    Greeting("Android")
                }
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun Greeting(name: String) {
    val cameraPermissionState= rememberPermissionState(android.Manifest.permission.CAMERA)
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
        Text(text = "Hello $name!")
    }
}

@Preview(showBackground = true)
@Composable
fun DefaultPreview() {
    SetFinderTheme {
        Greeting("Android")
    }
}