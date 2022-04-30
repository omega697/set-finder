package com.guywithburrito.setfinder

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionRequired
import com.guywithburrito.setfinder.ui.theme.SetFinderTheme
import com.google.accompanist.permissions.rememberPermissionState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import java.lang.RuntimeException

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SetFinderTheme {
                // A surface container using the 'background' color from the theme
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colors.background) {
                    Box(modifier = Modifier) {
                        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                        PermissionPreview(cameraSelector)
                        FloatingActionButton(
                            modifier = Modifier
                                .wrapContentSize()
                                .padding(16.dp)
                                .align(Alignment.BottomEnd),
                            onClick = {
                                Log.d("MainActivity", "TODO: Change camera.")
                            }
                        ) {
                            Icon(imageVector = Icons.Default.Cameraswitch,
                                contentDescription = "Switch cameras",
                                modifier = Modifier,
                            )
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun PermissionPreview(cameraSelector: CameraSelector) {
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
        CameraPreview(cameraSelector = cameraSelector)
    }
}