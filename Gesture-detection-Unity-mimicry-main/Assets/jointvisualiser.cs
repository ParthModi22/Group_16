using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System;

public class JointBot : MonoBehaviour
{
    UdpClient client;
    public int port = 5052;

    public float scale = 5f;
    public float boneLength = 0.8f;
    public float boneThickness = 0.08f;

    Vector3[] joints;
    Transform[] bones;
    GameObject headSphere;

    string lastData;

    [Serializable]
    public class Landmark
    {
        public float x, y, z, v;
    }

    int[,] chains = new int[,]
    {
        {11,13,15}, // left arm
        {12,14,16}, // right arm
        {23,25,27}, // left leg
        {24,26,28}  // right leg
    };

    int[,] extraConnections = new int[,]
    {
        {11,12},
        {23,24},
        {11,23},
        {12,24}
    };

    void Start()
    {
        client = new UdpClient(port);
        client.BeginReceive(ReceiveData, null);

        joints = new Vector3[33];

        int totalBones = chains.GetLength(0) * 2 + extraConnections.GetLength(0);
        bones = new Transform[totalBones];

        for (int i = 0; i < bones.Length; i++)
        {
            GameObject b = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            b.transform.localScale = new Vector3(boneThickness, boneLength / 2f, boneThickness);
            b.GetComponent<Renderer>().material.color = Color.green;
            bones[i] = b.transform;
        }

        headSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        headSphere.transform.localScale = Vector3.one * 0.4f;
    }

    void ReceiveData(IAsyncResult result)
    {
        IPEndPoint ip = new IPEndPoint(IPAddress.Any, port);
        byte[] data = client.EndReceive(result, ref ip);

        lastData = Encoding.UTF8.GetString(data);
        client.BeginReceive(ReceiveData, null);
    }

    void Update()
    {
        if (string.IsNullOrEmpty(lastData)) return;

        Landmark[] lm = JsonHelper.FromJson<Landmark>(lastData);
        if (lm == null) return;

        for (int i = 0; i < 33; i++)
            joints[i] = ToUnity(lm[i]) * scale;

        int idx = 0;

        for (int i = 0; i < chains.GetLength(0); i++)
        {
            int a = chains[i, 0];
            int b = chains[i, 1];
            int c = chains[i, 2];

            Vector3 A = joints[a];
            Vector3 B = joints[b];
            Vector3 C = joints[c];

            Vector3 BA = (A - B).normalized;
            Vector3 BC = (C - B).normalized;

            float angle = Vector3.Angle(BA, BC);

            // 🔥 Different limits for arms vs legs
            float minAngle = 5f;
            float maxAngle;

            if (i < 2) // arms (elbows)
                maxAngle = 180f;
            else        // legs (knees)
                maxAngle = 150f;

            float clamped = Mathf.Clamp(angle, minAngle, maxAngle);

            // Debug if needed
            // Debug.Log($"Joint {i} angle: {clamped}");

            DrawBone(bones[idx], A, B); idx++;
            DrawBone(bones[idx], B, C); idx++;
        }

        for (int i = 0; i < extraConnections.GetLength(0); i++)
        {
            DrawBone(bones[idx],
                joints[extraConnections[i, 0]],
                joints[extraConnections[i, 1]]);
            idx++;
        }

        headSphere.transform.position =
            (joints[0] + joints[7] + joints[8]) / 3f;
    }

    void DrawBone(Transform bone, Vector3 start, Vector3 end)
    {
        Vector3 mid = (start + end) * 0.5f;
        Vector3 dir = end - start;

        bone.position = mid;
        bone.up = dir.normalized;

        bone.localScale = new Vector3(
            boneThickness,
            dir.magnitude / 2f,
            boneThickness
        );
    }

    Vector3 ToUnity(Landmark lm)
    {
        return new Vector3(
            lm.x - 0.5f,
            -(lm.y - 0.5f),
            -lm.z
        );
    }
}