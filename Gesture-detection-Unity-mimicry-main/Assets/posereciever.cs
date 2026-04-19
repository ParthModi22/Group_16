using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System;

public class HumanoidDriver : MonoBehaviour
{
    public GameObject yBotPrefab;
    public int port = 5052;

    UdpClient client;
    string lastData;

    GameObject character;
    Animator animator;

    // Arms
    Transform leftUpperArm, leftLowerArm;
    Transform rightUpperArm, rightLowerArm;

    // Legs
    Transform leftUpperLeg, leftLowerLeg;
    Transform rightUpperLeg, rightLowerLeg;

    // Torso + head
    Transform hips, spine, chest, head;

    // Initial directions
    Vector3 lUpperInitDir, lLowerInitDir;
    Vector3 rUpperInitDir, rLowerInitDir;
    Vector3 lLegUpperInitDir, lLegLowerInitDir;
    Vector3 rLegUpperInitDir, rLegLowerInitDir;

    // Initial rotations
    Quaternion lUpperInitRot, lLowerInitRot;
    Quaternion rUpperInitRot, rLowerInitRot;
    Quaternion lLegUpperInitRot, lLegLowerInitRot;
    Quaternion rLegUpperInitRot, rLegLowerInitRot;

    Quaternion hipsInitRot, spineInitRot, chestInitRot, headInitRot;

    Vector3[] joints = new Vector3[33];

    Vector3 smoothedPosition;

    [Serializable]
    public class Landmark
    {
        public float x, y, z, v;
    }

    void Start()
    {
        client = new UdpClient(port);
        client.BeginReceive(ReceiveData, null);

        character = Instantiate(
            yBotPrefab,
            Vector3.zero,
            Quaternion.Euler(0, 0f, 0)
        );

        animator = character.GetComponent<Animator>();
        animator.enabled = false;

        // Bones
        leftUpperArm = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        leftLowerArm = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);

        rightUpperArm = animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
        rightLowerArm = animator.GetBoneTransform(HumanBodyBones.RightLowerArm);

        leftUpperLeg = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        leftLowerLeg = animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg);

        rightUpperLeg = animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        rightLowerLeg = animator.GetBoneTransform(HumanBodyBones.RightLowerLeg);

        hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        spine = animator.GetBoneTransform(HumanBodyBones.Spine);
        chest = animator.GetBoneTransform(HumanBodyBones.Chest);
        head = animator.GetBoneTransform(HumanBodyBones.Head);

        // Initial rotations
        lUpperInitRot = leftUpperArm.rotation;
        lLowerInitRot = leftLowerArm.rotation;
        rUpperInitRot = rightUpperArm.rotation;
        rLowerInitRot = rightLowerArm.rotation;

        lLegUpperInitRot = leftUpperLeg.rotation;
        lLegLowerInitRot = leftLowerLeg.rotation;
        rLegUpperInitRot = rightUpperLeg.rotation;
        rLegLowerInitRot = rightLowerLeg.rotation;

        hipsInitRot = hips.rotation;
        spineInitRot = spine.rotation;
        chestInitRot = chest.rotation;
        headInitRot = head.rotation;

        // Initial directions
        lUpperInitDir = (leftLowerArm.position - leftUpperArm.position).normalized;
        lLowerInitDir = (animator.GetBoneTransform(HumanBodyBones.LeftHand).position - leftLowerArm.position).normalized;

        rUpperInitDir = (rightLowerArm.position - rightUpperArm.position).normalized;
        rLowerInitDir = (animator.GetBoneTransform(HumanBodyBones.RightHand).position - rightLowerArm.position).normalized;

        lLegUpperInitDir = (leftLowerLeg.position - leftUpperLeg.position).normalized;
        lLegLowerInitDir = (animator.GetBoneTransform(HumanBodyBones.LeftFoot).position - leftLowerLeg.position).normalized;

        rLegUpperInitDir = (rightLowerLeg.position - rightUpperLeg.position).normalized;
        rLegLowerInitDir = (animator.GetBoneTransform(HumanBodyBones.RightFoot).position - rightLowerLeg.position).normalized;
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
        if (lm == null || lm.Length < 29) return;

        // Convert coords
        for (int i = 0; i < 33; i++)
        {
            joints[i] = new Vector3(
                (lm[i].x - 0.5f),
                -(lm[i].y - 0.5f),
                lm[i].z
            ) * 5f;
        }

        // -------------------------
        // LEG ACTIVITY
        // -------------------------
        float legActivity =
            (joints[25] - joints[23]).magnitude +
            (joints[26] - joints[24]).magnitude;

        legActivity *= 0.5f;

        float movementScale = Mathf.Lerp(0.5f, 0.9f, legActivity);

        // -------------------------
        // ROOT TARGET (JUMP ENABLED)
        // -------------------------
        Vector3 hipCenter = (joints[23] + joints[24]) * 0.5f;

        float verticalScale = 0.9f;

        float smoothedY = Mathf.Lerp(
            smoothedPosition.y,
            hipCenter.y * verticalScale,
            0.3f
        );

        Vector3 targetPos = new Vector3(
            hipCenter.x * movementScale,
            smoothedY,
            hipCenter.z * movementScale
        );

        // Dead zone
        if ((targetPos - smoothedPosition).magnitude < 0.02f)
        {
            targetPos = smoothedPosition;
        }

        // Velocity clamp
        float maxSpeed = 1f;

        Vector3 delta = targetPos - smoothedPosition;
        delta = Vector3.ClampMagnitude(delta, maxSpeed * Time.deltaTime);

        smoothedPosition += delta;

        character.transform.position = smoothedPosition;

        // Center joints
        for (int i = 0; i < joints.Length; i++)
        {
            joints[i] -= hipCenter;
        }

        // Body axes
        Vector3 leftShoulder = joints[11];
        Vector3 rightShoulder = joints[12];
        Vector3 shoulderMid = (leftShoulder + rightShoulder) * 0.5f;

        Vector3 bodyUp = shoulderMid.normalized;
        Vector3 bodyRight = (rightShoulder - leftShoulder).normalized;
        Vector3 bodyForward = Vector3.Cross(bodyRight, bodyUp).normalized;

        // Torso
        hips.rotation = Quaternion.LookRotation(bodyForward, bodyUp) * hipsInitRot;

        spine.rotation = Quaternion.Slerp(
            spine.rotation,
            Quaternion.LookRotation(bodyForward, bodyUp) * spineInitRot,
            0.3f
        );

        chest.rotation = Quaternion.Slerp(
            chest.rotation,
            Quaternion.LookRotation(bodyForward, bodyUp) * chestInitRot,
            0.5f
        );

        // Head
        Vector3 nose = joints[0];
        Vector3 headDir = (nose - shoulderMid).normalized;
        head.rotation = Quaternion.LookRotation(headDir, bodyUp) * headInitRot;

        // Arms
        Vector3 lUpperDir = (joints[13] - joints[11]).normalized;
        Vector3 lLowerDir = (joints[15] - joints[13]).normalized;

        leftUpperArm.rotation =
            Quaternion.FromToRotation(lUpperInitDir, lUpperDir) * lUpperInitRot;

        leftLowerArm.rotation =
            Quaternion.FromToRotation(lLowerInitDir, lLowerDir) * lLowerInitRot;

        Vector3 rUpperDir = (joints[14] - joints[12]).normalized;
        Vector3 rLowerDir = (joints[16] - joints[14]).normalized;

        rightUpperArm.rotation =
            Quaternion.FromToRotation(rUpperInitDir, rUpperDir) * rUpperInitRot;

        rightLowerArm.rotation =
            Quaternion.FromToRotation(rLowerInitDir, rLowerDir) * rLowerInitRot;

        // Legs
        Vector3 lLegUpperDir = (joints[25] - joints[23]) * 2f;
        Vector3 lLegLowerDir = (joints[27] - joints[25]) * 2f;

        lLegUpperDir.Normalize();
        lLegLowerDir.Normalize();

        leftUpperLeg.rotation =
            Quaternion.FromToRotation(lLegUpperInitDir, lLegUpperDir) * lLegUpperInitRot;

        leftLowerLeg.rotation =
            Quaternion.FromToRotation(lLegLowerInitDir, lLegLowerDir) * lLegLowerInitRot;

        Vector3 rLegUpperDir = (joints[26] - joints[24]) * 2f;
        Vector3 rLegLowerDir = (joints[28] - joints[26]) * 2f;

        rLegUpperDir.Normalize();
        rLegLowerDir.Normalize();

        rightUpperLeg.rotation =
            Quaternion.FromToRotation(rLegUpperInitDir, rLegUpperDir) * rLegUpperInitRot;

        rightLowerLeg.rotation =
            Quaternion.FromToRotation(rLegLowerInitDir, rLegLowerDir) * rLegLowerInitRot;
    }
}