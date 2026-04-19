using UnityEngine;

public class SpawnSpheresOnCube : MonoBehaviour
{
    public GameObject spherePrefab; // Assign a sphere prefab in Inspector
    public int numberOfSpheres = 33;

    void Start()
    {
        SpawnSpheres();
    }

    void SpawnSpheres()
    {
        Vector3 cubeCenter = transform.position;
        Vector3 cubeSize = transform.localScale;

        for (int i = 0; i < numberOfSpheres; i++)
        {
            Vector3 randomPoint = GetRandomPointOnCube(cubeCenter, cubeSize);
            Instantiate(spherePrefab, randomPoint, Quaternion.identity);
        }
    }

    Vector3 GetRandomPointOnCube(Vector3 center, Vector3 size)
    {
        // Pick one of the 6 faces randomly
        int face = Random.Range(0, 6);

        float x = Random.Range(-size.x / 2, size.x / 2);
        float y = Random.Range(-size.y / 2, size.y / 2);
        float z = Random.Range(-size.z / 2, size.z / 2);

        switch (face)
        {
            case 0: return center + new Vector3(size.x / 2, y, z); // Right
            case 1: return center + new Vector3(-size.x / 2, y, z); // Left
            case 2: return center + new Vector3(x, size.y / 2, z); // Top
            case 3: return center + new Vector3(x, -size.y / 2, z); // Bottom
            case 4: return center + new Vector3(x, y, size.z / 2); // Front
            case 5: return center + new Vector3(x, y, -size.z / 2); // Back
            default: return center;
        }
    }
}